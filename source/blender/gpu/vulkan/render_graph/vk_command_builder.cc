/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "vk_command_builder.hh"
#include "BLI_index_range.hh"
#include "vk_backend.hh"
#include "vk_render_graph.hh"
#include "vk_to_string.hh"

#include <sstream>

namespace blender::gpu::render_graph {

static VkImageLayout to_default_image_layout(VkImageAspectFlags aspect, bool use_local_read)
{
  if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
    if (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) {
      return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    return vk_image_layout_supported(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  }
  if (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) {
    return vk_image_layout_supported(VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
  }
  return use_local_read ? VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR :
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
}

/* -------------------------------------------------------------------- */
/** \name Build nodes
 * \{ */

void VKCommandBuilder::build_nodes(VKRenderGraph &render_graph,
                                   VKCommandBufferInterface &command_buffer,
                                   Span<NodeHandle> node_handles)
{
  groups_init(render_graph, node_handles);
  groups_extract_barriers(
      render_graph, node_handles, command_buffer.use_dynamic_rendering_local_read);
}

void VKCommandBuilder::record_commands(VKRenderGraph &render_graph,
                                       VKCommandBufferInterface &command_buffer,
                                       Span<NodeHandle> node_handles)
{
  groups_build_commands(render_graph, command_buffer, node_handles);
}

void VKCommandBuilder::groups_init(const VKRenderGraph &render_graph,
                                   Span<NodeHandle> node_handles)
{
  group_nodes_.clear();
  IndexRange nodes_range = node_handles.index_range();
  while (!nodes_range.is_empty()) {
    IndexRange node_group = nodes_range.slice(0, 1);
    NodeHandle node_handle = node_handles[nodes_range.first()];
    const VKRenderGraphNode &node = render_graph.nodes_[node_handle];
    while (node_type_is_rendering(node.type) && node_group.size() < nodes_range.size()) {
      NodeHandle node_handle = node_handles[nodes_range[node_group.size()]];
      const VKRenderGraphNode &node = render_graph.nodes_[node_handle];
      if (!node_type_is_rendering(node.type) || node.type == VKNodeType::BEGIN_RENDERING) {
        break;
      }
      node_group = nodes_range.slice(0, node_group.size() + 1);
    }

    group_nodes_.append(node_group);
    nodes_range = nodes_range.drop_front(node_group.size());
  }
}

void VKCommandBuilder::groups_extract_barriers(VKRenderGraph &render_graph,
                                               Span<NodeHandle> node_handles,
                                               bool use_local_read)
{
  barrier_list_.clear();
  vk_buffer_memory_barriers_.clear();
  vk_image_memory_barriers_.clear();

  ImageTracker image_tracker(*this);

  /* Extract barriers. */
  group_pre_barriers_.clear();
  group_post_barriers_.clear();
  node_pre_barriers_.resize(node_handles.size());

  Vector<Barrier> post_barriers;
  Vector<Barrier> node_pre_barriers;

  NodeHandle rendering_scope;
  bool rendering_active = false;

  for (const int64_t group_index : group_nodes_.index_range()) {
    Barriers group_pre_barriers(barrier_list_.size(), 0);
    const GroupNodes &node_group = group_nodes_[group_index];
    for (const int64_t group_node_index : node_group) {
      NodeHandle node_handle = node_handles[group_node_index];
      VKRenderGraphNode &node = render_graph.nodes_[node_handle];
      Barrier barrier = {};
      build_pipeline_barriers(
          render_graph, node_handle, node.pipeline_stage_get(), image_tracker, barrier);
      if (!barrier.is_empty()) {
        barrier_list_.append(barrier);
      }

      if (node.type == VKNodeType::SYNCHRONIZATION &&
          node.synchronization.reset_layout_to_undefined)
      {
        for (const VKRenderGraphImage &link : render_graph.linked_images(node_handle)) {
          VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
              link.resource.handle);
          resource.barrier_state.image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
      }
      else if (node.type == VKNodeType::BEGIN_RENDERING) {
        BLI_assert(!rendering_active);
        rendering_scope = node_handle;
        rendering_active = true;
        image_tracker.begin(render_graph, node_handle);
      }
      else if (node.type == VKNodeType::END_RENDERING) {
        BLI_assert(rendering_active);

        const NodeHandle end_rendering_scope = rendering_scope;
        rendering_scope = 0;
        rendering_active = false;

        for (const VKRenderGraphImage &link : render_graph.linked_images(end_rendering_scope)) {
          VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
              link.resource.handle);
          VKResourceBarrierState &state = resource.barrier_state;
          if (link.vk_access_flags & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) {
            state.vk_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          }
          if (link.vk_access_flags & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) {
            state.vk_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
          }
        }

        Barrier barrier = {};
        image_tracker.end(barrier, use_local_read);
        if (!barrier.is_empty()) {
          post_barriers.append(barrier);
        }

        for (const VKRenderGraphImage &link : render_graph.linked_images(end_rendering_scope)) {
          VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
              link.resource.handle);
          if (resource.use_subresource_tracking()) {
            VKResourceBarrierState &state = resource.barrier_state;
            state.image_layout = to_default_image_layout(link.vk_image_aspect, use_local_read);
          }
        }
      }
      else if (rendering_active && !node_type_is_within_rendering(node.type)) {
        rendering_active = false;

        Barrier barrier = {};
        image_tracker.suspend(barrier, use_local_read);
        if (!barrier.is_empty()) {
          post_barriers.append(barrier);
        }

        for (const VKRenderGraphImage &link : render_graph.linked_images(rendering_scope)) {
          VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
              link.resource.handle);
          if (resource.use_subresource_tracking()) {
            VKResourceBarrierState &state = resource.barrier_state;
            state.image_layout = to_default_image_layout(link.vk_image_aspect, use_local_read);
          }
        }
      }
      else if (!rendering_active && node_type_is_within_rendering(node.type)) {
        VKRenderGraphNode &rendering_node = render_graph.nodes_[rendering_scope];
        Barrier barrier = {};
        build_pipeline_barriers(render_graph,
                                rendering_scope,
                                rendering_node.pipeline_stage_get(),
                                image_tracker,
                                barrier);
        if (!barrier.is_empty()) {
          barrier_list_.append(barrier);
        }

        barrier = {};
        image_tracker.resume(barrier, use_local_read);
        if (!barrier.is_empty()) {
          barrier_list_.append(barrier);
        }

        rendering_active = true;
      }

      if (use_local_read && node_type_is_within_rendering(node.type) &&
          node_has_input_attachments(render_graph, node_handle))
      {
        Barrier barrier = {};
        build_pipeline_barriers(
            render_graph, node_handle, node.pipeline_stage_get(), image_tracker, barrier, true);
        if (!barrier.is_empty()) {
          node_pre_barriers.append(barrier);
        }
      }
    }
    if (rendering_active) {
      Barrier barrier = {};
      image_tracker.suspend(barrier, use_local_read);
      if (!barrier.is_empty()) {
        post_barriers.append(barrier);
      }

      for (const VKRenderGraphImage &link : render_graph.linked_images(rendering_scope)) {
        VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
            link.resource.handle);
        if (resource.use_subresource_tracking()) {
          VKResourceBarrierState &state = resource.barrier_state;
          state.image_layout = to_default_image_layout(link.vk_image_aspect, use_local_read);
        }
      }

      rendering_active = false;
    }

    {
      NodeHandle first_handle = node_handles[node_group.first()];
      VKRenderGraphNode &first_node = render_graph.nodes_[first_handle];
      if (first_node.type == VKNodeType::BEGIN_RENDERING) {
        VKBeginRenderingData &begin_data =
            render_graph.storage_.begin_rendering[first_node.storage_index];

        for (const VKRenderGraphImage &link : render_graph.linked_images(first_handle)) {
          if (!(link.vk_image_aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)))
          {
            continue;
          }
          VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
              link.resource.handle);
          VkImageLayout gpu_layout = resource.barrier_state.image_layout;
          VkImageLayout optimal_layout = to_default_image_layout(link.vk_image_aspect,
                                                                 use_local_read);
          if (gpu_layout == optimal_layout) {
            continue;
          }

          if (link.vk_image_aspect & VK_IMAGE_ASPECT_DEPTH_BIT &&
              begin_data.vk_rendering_info.pDepthAttachment != nullptr)
          {
            begin_data.depth_attachment.imageLayout = gpu_layout;
          }
          if (link.vk_image_aspect & VK_IMAGE_ASPECT_STENCIL_BIT &&
              begin_data.vk_rendering_info.pStencilAttachment != nullptr)
          {
            begin_data.stencil_attachment.imageLayout = gpu_layout;
          }

          bool depth_load = (link.vk_image_aspect & VK_IMAGE_ASPECT_DEPTH_BIT) &&
                            begin_data.vk_rendering_info.pDepthAttachment != nullptr &&
                            begin_data.depth_attachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
          bool stencil_load = (link.vk_image_aspect & VK_IMAGE_ASPECT_STENCIL_BIT) &&
                              begin_data.vk_rendering_info.pStencilAttachment != nullptr &&
                              begin_data.stencil_attachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
          if (!depth_load && !stencil_load) {
            continue;
          }

          Barrier barrier = {};
          reset_barriers(barrier);
          barrier.src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
          barrier.dst_stage_mask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
          barrier.image_memory_barriers = IndexRange(vk_image_memory_barriers_.size(), 0);
          add_image_barrier(resource.image.vk_image,
                            barrier,
                            VK_ACCESS_MEMORY_WRITE_BIT,
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                            gpu_layout,
                            gpu_layout,
                            link.vk_image_aspect,
                            {});
          barrier.image_memory_barriers = barrier.image_memory_barriers.with_new_end(
              vk_image_memory_barriers_.size());
          barrier_list_.append(barrier);
        }
      }
    }

    int64_t barrier_list_size = barrier_list_.size();
    group_pre_barriers_.append(group_pre_barriers.with_new_end(barrier_list_size));
    barrier_list_.extend(std::move(post_barriers));
    post_barriers = {};
    group_post_barriers_.append(
        IndexRange::from_begin_end(barrier_list_size, barrier_list_.size()));
    if (!node_pre_barriers.is_empty()) {
      barrier_list_size = barrier_list_.size();
      barrier_list_.extend(std::move(node_pre_barriers));
      node_pre_barriers = {};
      for (const int64_t group_node_index : node_group) {
        NodeHandle node_handle = node_handles[group_node_index];
        if (!node_pre_barriers_[node_handle].is_empty()) {
          node_pre_barriers_[node_handle] = IndexRange::from_begin_size(
              node_pre_barriers_[node_handle].start() + barrier_list_size, 1);
        }
      }
    }
  }

  BLI_assert(group_pre_barriers_.size() == group_nodes_.size());
  BLI_assert(group_post_barriers_.size() == group_nodes_.size());
}

void VKCommandBuilder::groups_build_commands(VKRenderGraph &render_graph,
                                             VKCommandBufferInterface &command_buffer,
                                             Span<NodeHandle> node_handles)
{
  DebugGroups debug_groups = {};
  VKBoundPipelines active_pipelines = {};

  NodeHandle rendering_scope = 0;
  bool rendering_active = false;

  for (int64_t group_index : group_nodes_.index_range()) {
    IndexRange group_nodes = group_nodes_[group_index];
    Span<NodeHandle> group_node_handles = node_handles.slice(group_nodes);

    for (BarrierIndex barrier_index : group_pre_barriers_[group_index]) {
      BLI_assert_msg(!rendering_active,
                     "Pre group barriers must be executed outside a rendering scope.");
      Barrier &barrier = barrier_list_[barrier_index];
      send_pipeline_barriers(command_buffer, barrier, false);
    }

    for (NodeHandle node_handle : group_node_handles) {
      VKRenderGraphNode &node = render_graph.nodes_[node_handle];

      if (G.debug & G_DEBUG_GPU) {
        activate_debug_group(render_graph, command_buffer, debug_groups, node_handle);
      }

      if (node.type == VKNodeType::BEGIN_RENDERING) {
        rendering_scope = node_handle;
        rendering_active = true;
      }
      else if (node.type == VKNodeType::END_RENDERING) {
        rendering_active = false;
      }
      else if (node_type_is_within_rendering(node.type)) {
        if (!rendering_active) {
          VKRenderGraphNode &rendering_node = render_graph.nodes_[rendering_scope];
          VKBeginRenderingNode::reconfigure_for_restart(
              render_graph.storage_.begin_rendering[rendering_node.storage_index]);
          rendering_node.build_commands(command_buffer, render_graph.storage_, active_pipelines);
          rendering_active = true;
        }
      }

      for (BarrierIndex node_pre_barrier_index : node_pre_barriers_[node_handle]) {
        Barrier &barrier = barrier_list_[node_pre_barrier_index];
        send_pipeline_barriers(command_buffer, barrier, true);
      }

      node.build_commands(command_buffer, render_graph.storage_, active_pipelines);
    }

    if (rendering_active) {
      rendering_active = false;
      if (command_buffer.use_render_pass_fallback) {
        command_buffer.end_render_pass();
      }
      else {
        command_buffer.end_rendering();
      }
    }

    for (BarrierIndex barrier_index : group_post_barriers_[group_index]) {
      BLI_assert_msg(!rendering_active,
                     "Post group barriers must be executed outside a rendering scope.");
      Barrier &barrier = barrier_list_[barrier_index];
      send_pipeline_barriers(command_buffer, barrier, false);
    }
  }

  finish_debug_groups(command_buffer, debug_groups);
}

bool VKCommandBuilder::node_has_input_attachments(const VKRenderGraph &render_graph,
                                                  NodeHandle node_handle)
{
  const Span<VKRenderGraphImage> images = render_graph.linked_images(node_handle);
  return std::ranges::any_of(images.begin(), images.end(), [](const VKRenderGraphImage &input) {
    return input.vk_access_flags & VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
  });
}

void VKCommandBuilder::activate_debug_group(VKRenderGraph &render_graph,
                                            VKCommandBufferInterface &command_buffer,
                                            DebugGroups &debug_groups,
                                            NodeHandle node_handle)
{
  VKRenderGraph::DebugGroupID debug_group = render_graph.debug_.node_group_map[node_handle];
  if (debug_group == debug_groups.active_debug_group_id) {
    return;
  }

  int num_ends = 0;
  int num_begins = 0;

  if (debug_group == -1) {
    num_ends = debug_groups.debug_level;
  }
  else {
    Vector<VKRenderGraph::DebugGroupNameID> &to_group =
        render_graph.debug_.used_groups[debug_group];
    if (debug_groups.active_debug_group_id != -1) {
      Vector<VKRenderGraph::DebugGroupNameID> &from_group =
          render_graph.debug_.used_groups[debug_groups.active_debug_group_id];

      num_ends = max_ii(from_group.size() - to_group.size(), 0);
      int num_checks = min_ii(from_group.size(), to_group.size());
      for (int index : IndexRange(num_checks)) {
        if (from_group[index] != to_group[index]) {
          num_ends += num_checks - index;
          break;
        }
      }
    }

    num_begins = to_group.size() - (debug_groups.debug_level - num_ends);
  }

  for (int index = 0; index < num_ends; index++) {
    command_buffer.end_debug_utils_label();
  }
  debug_groups.debug_level -= num_ends;

  if (num_begins > 0) {
    Vector<VKRenderGraph::DebugGroupNameID> &to_group =
        render_graph.debug_.used_groups[debug_group];
    VkDebugUtilsLabelEXT debug_utils_label = {};
    debug_utils_label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    for (int index : IndexRange(debug_groups.debug_level, num_begins)) {
      const VKRenderGraph::DebugGroup &debug_group = render_graph.debug_.groups[to_group[index]];
      debug_utils_label.pLabelName = debug_group.name.c_str();
      copy_v4_v4(debug_utils_label.color, debug_group.color);
      command_buffer.begin_debug_utils_label(&debug_utils_label);
    }
  }

  debug_groups.debug_level += num_begins;
  debug_groups.active_debug_group_id = debug_group;
}

void VKCommandBuilder::finish_debug_groups(VKCommandBufferInterface &command_buffer,
                                           DebugGroups &debug_groups)
{
  for (int i = 0; i < debug_groups.debug_level; i++) {
    command_buffer.end_debug_utils_label();
  }
  debug_groups.debug_level = 0;
}

void VKCommandBuilder::build_pipeline_barriers(VKRenderGraph &render_graph,
                                               NodeHandle node_handle,
                                               VkPipelineStageFlags pipeline_stage,
                                               ImageTracker &image_tracker,
                                               Barrier &r_barrier,
                                               bool within_rendering)
{
  reset_barriers(r_barrier);
  add_image_barriers(
      render_graph, node_handle, pipeline_stage, image_tracker, r_barrier, within_rendering);
  add_buffer_barriers(render_graph, node_handle, pipeline_stage, r_barrier);
}

void VKCommandBuilder::reset_barriers(Barrier &r_barrier)
{
  r_barrier.dst_stage_mask = r_barrier.src_stage_mask = VK_PIPELINE_STAGE_NONE;
}

void VKCommandBuilder::send_pipeline_barriers(VKCommandBufferInterface &command_buffer,
                                              const Barrier &barrier,
                                              bool within_rendering)
{
  if (barrier.is_empty()) {
    return;
  }

  VkPipelineStageFlags src_stage_mask = (barrier.src_stage_mask == VK_PIPELINE_STAGE_NONE) ?
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
                                            VkPipelineStageFlagBits(barrier.src_stage_mask);

  VkPipelineStageFlags dst_stage_mask = barrier.dst_stage_mask;
  if (within_rendering) {
    src_stage_mask = dst_stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }

  Span<VkBufferMemoryBarrier> buffer_barriers = vk_buffer_memory_barriers_.as_span().slice(
      barrier.buffer_memory_barriers);
  Span<VkImageMemoryBarrier> image_barriers = vk_image_memory_barriers_.as_span().slice(
      barrier.image_memory_barriers);

  command_buffer.pipeline_barrier(src_stage_mask,
                                  dst_stage_mask,
                                  VK_DEPENDENCY_BY_REGION_BIT,
                                  0,
                                  nullptr,
                                  buffer_barriers.size(),
                                  buffer_barriers.data(),
                                  image_barriers.size(),
                                  image_barriers.data());
}

void VKCommandBuilder::add_buffer_barriers(VKRenderGraph &render_graph,
                                           NodeHandle node_handle,
                                           VkPipelineStageFlags node_stages,
                                           Barrier &r_barrier)
{
  r_barrier.buffer_memory_barriers = IndexRange(vk_buffer_memory_barriers_.size(), 0);
  add_buffer_read_barriers(render_graph, node_handle, node_stages, r_barrier);
  add_buffer_write_barriers(render_graph, node_handle, node_stages, r_barrier);
  r_barrier.buffer_memory_barriers = r_barrier.buffer_memory_barriers.with_new_end(
      vk_buffer_memory_barriers_.size());
}

void VKCommandBuilder::add_buffer_read_barriers(VKRenderGraph &render_graph,
                                                NodeHandle node_handle,
                                                VkPipelineStageFlags node_stages,
                                                Barrier &r_barrier)
{
  for (const VKRenderGraphBuffer &link : render_graph.linked_buffers(node_handle)) {
    if (link.has_write_access()) {
      continue;
    }
    const ResourceWithStamp &versioned_resource = link.resource;
    VKResourceStateTracker::Resource &resource = render_graph.resources_.get_buffer_resource(
        versioned_resource.handle);
    VKResourceBarrierState &resource_state = resource.barrier_state;
    const bool is_first_read = resource_state.is_new_stamp();
    if (!is_first_read &&
        (resource_state.vk_access & link.vk_access_flags) == link.vk_access_flags &&
        (resource_state.vk_pipeline_stages & node_stages) == node_stages)
    {
      continue;
    }

    const VkAccessFlags wait_access = resource_state.vk_access;

    r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
    r_barrier.dst_stage_mask |= node_stages;

    if (is_first_read) {
      resource_state.vk_access = link.vk_access_flags;
      resource_state.vk_pipeline_stages = node_stages;
    }
    else {
      resource_state.vk_access |= link.vk_access_flags;
      resource_state.vk_pipeline_stages |= node_stages;
    }

    if (wait_access != VK_ACCESS_NONE) {
      add_buffer_barrier(resource.buffer.vk_buffer, r_barrier, wait_access, link.vk_access_flags);
    }
  }
}

void VKCommandBuilder::add_buffer_write_barriers(VKRenderGraph &render_graph,
                                                 NodeHandle node_handle,
                                                 VkPipelineStageFlags node_stages,
                                                 Barrier &r_barrier)
{
  for (const VKRenderGraphBuffer &link : render_graph.linked_buffers(node_handle)) {
    if (!link.has_write_access()) {
      continue;
    }
    const ResourceWithStamp &versioned_resource = link.resource;
    VKResourceStateTracker::Resource &resource = render_graph.resources_.get_buffer_resource(
        versioned_resource.handle);
    VKResourceBarrierState &resource_state = resource.barrier_state;
    const VkAccessFlags wait_access = resource_state.vk_access;

    r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
    r_barrier.dst_stage_mask |= node_stages;

    resource_state.vk_access = link.vk_access_flags;
    resource_state.vk_pipeline_stages = node_stages;

    if (wait_access != VK_ACCESS_NONE) {
      add_buffer_barrier(resource.buffer.vk_buffer, r_barrier, wait_access, link.vk_access_flags);
    }
  }
}

void VKCommandBuilder::add_buffer_barrier(VkBuffer vk_buffer,
                                          Barrier &r_barrier,
                                          VkAccessFlags src_access_mask,
                                          VkAccessFlags dst_access_mask)
{
  for (VkBufferMemoryBarrier &vk_buffer_memory_barrier :
       vk_buffer_memory_barriers_.as_mutable_span().drop_front(
           r_barrier.buffer_memory_barriers.start()))
  {
    if (vk_buffer_memory_barrier.buffer == vk_buffer) {
      if ((vk_buffer_memory_barrier.dstAccessMask & src_access_mask) == src_access_mask) {
        vk_buffer_memory_barrier.dstAccessMask |= dst_access_mask;
        return;
      }
      if ((vk_buffer_memory_barrier.dstAccessMask & dst_access_mask) == dst_access_mask &&
          (vk_buffer_memory_barrier.srcAccessMask & src_access_mask) == src_access_mask)
      {
        return;
      }
    }
  }

  vk_buffer_memory_barriers_.append({VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                     nullptr,
                                     src_access_mask,
                                     dst_access_mask,
                                     VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED,
                                     vk_buffer,
                                     0,
                                     VK_WHOLE_SIZE});
}

void VKCommandBuilder::add_image_barriers(VKRenderGraph &render_graph,
                                          NodeHandle node_handle,
                                          VkPipelineStageFlags node_stages,
                                          ImageTracker &image_tracker,
                                          Barrier &r_barrier,
                                          bool within_rendering)
{
  r_barrier.image_memory_barriers = IndexRange(vk_image_memory_barriers_.size(), 0);
  add_image_read_barriers(
      render_graph, node_handle, node_stages, image_tracker, r_barrier, within_rendering);
  add_image_write_barriers(
      render_graph, node_handle, node_stages, image_tracker, r_barrier, within_rendering);
  r_barrier.image_memory_barriers = r_barrier.image_memory_barriers.with_new_end(
      vk_image_memory_barriers_.size());
}

void VKCommandBuilder::add_image_read_barriers(VKRenderGraph &render_graph,
                                               NodeHandle node_handle,
                                               VkPipelineStageFlags node_stages,
                                               ImageTracker &image_tracker,
                                               Barrier &r_barrier,
                                               bool within_rendering)
{
  for (const VKRenderGraphImage &link : render_graph.linked_images(node_handle)) {
    if (link.has_write_access()) {
      continue;
    }
    const ResourceWithStamp &versioned_resource = link.resource;
    VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
        versioned_resource.handle);
    VKResourceBarrierState &resource_state = resource.barrier_state;
    const bool is_first_read = resource_state.is_new_stamp();
    if ((!is_first_read) &&
        (resource_state.vk_access & link.vk_access_flags) == link.vk_access_flags &&
        (resource_state.vk_pipeline_stages & node_stages) == node_stages &&
        resource_state.image_layout == link.vk_image_layout)
    {
      continue;
    }
    if (within_rendering && link.vk_image_layout != VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR) {
      continue;
    }

    if (image_tracker.contains(resource.image.vk_image)) {
      if (resource_state.image_layout != link.vk_image_layout) {
        image_tracker.update(resource.image.vk_image,
                             link.subimage,
                             resource_state.image_layout,
                             link.vk_image_layout,
                             link.vk_image_aspect,
                             r_barrier);
      }
      VkAccessFlags wait_access = resource_state.vk_access;
      r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
      r_barrier.dst_stage_mask |= node_stages;
      if (is_first_read) {
        resource_state.vk_access = link.vk_access_flags;
        resource_state.vk_pipeline_stages = node_stages;
      }
      else {
        resource_state.vk_access |= link.vk_access_flags;
        resource_state.vk_pipeline_stages |= node_stages;
      }
      if (wait_access != VK_ACCESS_NONE && (wait_access & ~link.vk_access_flags) != 0) {
        add_image_barrier(resource.image.vk_image,
                          r_barrier,
                          wait_access,
                          link.vk_access_flags,
                          resource_state.image_layout,
                          link.vk_image_layout,
                          link.vk_image_aspect,
                          {});
      }
      continue;
    }

    VkAccessFlags wait_access = resource_state.vk_access;

    r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
    r_barrier.dst_stage_mask |= node_stages;

    if (is_first_read) {
      resource_state.vk_access = link.vk_access_flags;
      resource_state.vk_pipeline_stages = node_stages;
    }
    else {
      resource_state.vk_access |= link.vk_access_flags;
      resource_state.vk_pipeline_stages |= node_stages;
    }

    if (wait_access != VK_ACCESS_NONE || link.vk_image_layout != resource_state.image_layout) {
      add_image_barrier(resource.image.vk_image,
                        r_barrier,
                        wait_access,
                        link.vk_access_flags,
                        resource_state.image_layout,
                        link.vk_image_layout,
                        link.vk_image_aspect,
                        {});
      resource_state.image_layout = link.vk_image_layout;
    }
  }
}

void VKCommandBuilder::add_image_write_barriers(VKRenderGraph &render_graph,
                                                NodeHandle node_handle,
                                                VkPipelineStageFlags node_stages,
                                                ImageTracker &image_tracker,
                                                Barrier &r_barrier,
                                                bool within_rendering)
{
  for (const VKRenderGraphImage &link : render_graph.linked_images(node_handle)) {
    if (!link.has_write_access()) {
      continue;
    }
    const ResourceWithStamp &versioned_resource = link.resource;
    VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
        versioned_resource.handle);
    VKResourceBarrierState &resource_state = resource.barrier_state;
    const VkAccessFlags wait_access = resource_state.vk_access;
    if (within_rendering && link.vk_image_layout != VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR) {
      continue;
    }

    if (image_tracker.contains(resource.image.vk_image)) {
      if (resource_state.image_layout != link.vk_image_layout) {
        image_tracker.update(resource.image.vk_image,
                             link.subimage,
                             resource_state.image_layout,
                             link.vk_image_layout,
                             link.vk_image_aspect,
                             r_barrier);
      }
      r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
      r_barrier.dst_stage_mask |= node_stages;
      resource_state.vk_access = link.vk_access_flags;
      resource_state.vk_pipeline_stages = node_stages;
      if (wait_access != VK_ACCESS_NONE && (wait_access & ~link.vk_access_flags) != 0) {
        add_image_barrier(resource.image.vk_image,
                          r_barrier,
                          wait_access,
                          link.vk_access_flags,
                          resource_state.image_layout,
                          link.vk_image_layout,
                          link.vk_image_aspect,
                          {});
      }
      continue;
    }

    r_barrier.src_stage_mask |= resource_state.vk_pipeline_stages;
    r_barrier.dst_stage_mask |= node_stages;

    resource_state.vk_access = link.vk_access_flags;
    resource_state.vk_pipeline_stages = node_stages;

    if (wait_access != VK_ACCESS_NONE || link.vk_image_layout != resource_state.image_layout) {
      add_image_barrier(resource.image.vk_image,
                        r_barrier,
                        wait_access,
                        link.vk_access_flags,
                        resource_state.image_layout,
                        link.vk_image_layout,
                        link.vk_image_aspect,
                        {});
      resource_state.image_layout = link.vk_image_layout;
    }
  }
}

void VKCommandBuilder::add_image_barrier(VkImage vk_image,
                                         Barrier &r_barrier,
                                         VkAccessFlags src_access_mask,
                                         VkAccessFlags dst_access_mask,
                                         VkImageLayout old_image_layout,
                                         VkImageLayout new_image_layout,
                                         VkImageAspectFlags aspect_mask,
                                         const VKSubImageRange &subimage)
{
  BLI_assert(aspect_mask != VK_IMAGE_ASPECT_NONE);
  for (VkImageMemoryBarrier &vk_image_memory_barrier :
       vk_image_memory_barriers_.as_mutable_span().drop_front(
           r_barrier.image_memory_barriers.start()))
  {
    if (vk_image_memory_barrier.image == vk_image) {
      if ((vk_image_memory_barrier.dstAccessMask & src_access_mask) == src_access_mask) {
        vk_image_memory_barrier.dstAccessMask |= dst_access_mask;
        vk_image_memory_barrier.srcAccessMask |= src_access_mask;
        if (old_image_layout == vk_image_memory_barrier.newLayout) {
          vk_image_memory_barrier.newLayout = new_image_layout;
        }
        return;
      }
      if ((vk_image_memory_barrier.dstAccessMask & dst_access_mask) == dst_access_mask &&
          (vk_image_memory_barrier.srcAccessMask & src_access_mask) == src_access_mask &&
          old_image_layout == new_image_layout)
      {
        return;
      }
    }
  }

  vk_image_memory_barriers_.append({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    nullptr,
                                    src_access_mask,
                                    dst_access_mask,
                                    old_image_layout,
                                    new_image_layout,
                                    VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED,
                                    vk_image,
                                    {aspect_mask,
                                     subimage.mipmap_level,
                                     subimage.mipmap_count,
                                     subimage.layer_base,
                                     subimage.layer_count}});
}

void VKCommandBuilder::ImageTracker::begin(const VKRenderGraph &render_graph,
                                           NodeHandle node_handle)
{
  BLI_assert(render_graph.nodes_[node_handle].type == VKNodeType::BEGIN_RENDERING);
  tracked_attachments.clear();
  changes.clear();

  for (const VKRenderGraphImage &link : render_graph.linked_images(node_handle)) {
    VKResourceStateTracker::Resource &resource = render_graph.resources_.get_image_resource(
        link.resource.handle);
    if (resource.use_subresource_tracking()) {
      tracked_attachments.add(resource.image.vk_image);
    }
  }
}

void VKCommandBuilder::ImageTracker::update(VkImage vk_image,
                                            const VKSubImageRange &subimage,
                                            VkImageLayout old_layout,
                                            VkImageLayout new_layout,
                                            VkImageAspectFlags aspect,
                                            Barrier &r_barrier)
{
  for (const SubImageChange &change : changes) {
    if (change.vk_image == vk_image && ((subimage.layer_count != VK_REMAINING_ARRAY_LAYERS &&
                                         change.subimage.layer_base == subimage.layer_base) ||
                                        (subimage.mipmap_count != VK_REMAINING_MIP_LEVELS &&
                                         change.subimage.mipmap_level == subimage.mipmap_level)))
    {
      BLI_assert_msg(
          change.vk_image_layout == new_layout,
          "We don't support more that one change of the same subimage multiple times during a "
          "rendering scope.");
      return;
    }
  }

  changes.append({vk_image, new_layout, aspect, subimage});

  r_barrier.src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  r_barrier.dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkAccessFlags dst_access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  if (aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
    dst_access_mask |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  }
  if (aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
    dst_access_mask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  }
  command_builder.add_image_barrier(vk_image,
                                    r_barrier,
                                    VK_ACCESS_TRANSFER_WRITE_BIT,
                                    dst_access_mask,
                                    old_layout,
                                    new_layout,
                                    aspect,
                                    subimage);
}

void VKCommandBuilder::ImageTracker::end(Barrier &r_barrier, bool use_local_read)
{
  suspend(r_barrier, use_local_read);
  tracked_attachments.clear();
  changes.clear();
}

void VKCommandBuilder::ImageTracker::suspend(Barrier &r_barrier, bool use_local_read)
{
  if (changes.is_empty()) {
    return;
  }

  command_builder.reset_barriers(r_barrier);
  r_barrier.src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  r_barrier.dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  int64_t start_index = command_builder.vk_image_memory_barriers_.size();
  r_barrier.image_memory_barriers = IndexRange::from_begin_size(start_index, 0);
  for (const SubImageChange &change : changes) {
    command_builder.add_image_barrier(
        change.vk_image,
        r_barrier,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        change.vk_image_layout,
        to_default_image_layout(change.vk_image_aspect, use_local_read),
        change.vk_image_aspect,
        change.subimage);
    r_barrier.image_memory_barriers = r_barrier.image_memory_barriers.with_new_end(
        command_builder.vk_image_memory_barriers_.size());
  }
}

void VKCommandBuilder::ImageTracker::resume(Barrier &r_barrier, bool use_local_read)
{
  if (changes.is_empty()) {
    return;
  }

  command_builder.reset_barriers(r_barrier);
  r_barrier.src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  r_barrier.dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  int64_t start_index = command_builder.vk_image_memory_barriers_.size();
  r_barrier.image_memory_barriers = IndexRange::from_begin_size(start_index, 0);

  for (const SubImageChange &change : changes) {
    command_builder.add_image_barrier(
        change.vk_image,
        r_barrier,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        to_default_image_layout(change.vk_image_aspect, use_local_read),
        change.vk_image_layout,
        change.vk_image_aspect,
        change.subimage);
  }
}

std::string VKCommandBuilder::to_string_barrier(const Barrier &barrier)
{
  std::stringstream ss;
  ss << "src_stage_mask=" << to_string_vk_pipeline_stage_flags(barrier.src_stage_mask)
     << ", dst_stage_mask=" << to_string_vk_pipeline_stage_flags(barrier.dst_stage_mask) << "\n";
  for (const VkBufferMemoryBarrier &buffer_memory_barrier :
       vk_buffer_memory_barriers_.as_span().slice(barrier.buffer_memory_barriers))
  {
    ss << "  - src_access_mask=" << to_string_vk_access_flags(buffer_memory_barrier.srcAccessMask)
       << ", dst_access_mask=" << to_string_vk_access_flags(buffer_memory_barrier.dstAccessMask)
       << ", vk_buffer=" << to_string(buffer_memory_barrier.buffer) << "\n";
  }

  for (const VkImageMemoryBarrier &image_memory_barrier :
       vk_image_memory_barriers_.as_span().slice(barrier.image_memory_barriers))
  {
    ss << "  - src_access_mask=" << to_string_vk_access_flags(image_memory_barrier.srcAccessMask)
       << ", dst_access_mask=" << to_string_vk_access_flags(image_memory_barrier.dstAccessMask)
       << ", vk_image=" << to_string(image_memory_barrier.image)
       << ", old_layout=" << to_string(image_memory_barrier.oldLayout)
       << ", new_layout=" << to_string(image_memory_barrier.newLayout)
       << ", subresource_range=" << to_string(image_memory_barrier.subresourceRange, 2) << "\n";
  }

  return ss.str();
}

}  // namespace blender::gpu::render_graph
