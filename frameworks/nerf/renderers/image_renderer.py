import torch


class ImageRenderer:
    def __init__(self, batch_size=4096, key='rgb_marched', **kwargs):
        self.kwargs = kwargs
        self.bs = batch_size
        self.key = key

    def renderView(self, H, W, K, render_pose, nerf):
        """
        render a view by parameters
        :type render_pose:
        @param H: height of image
        @param W: width of image
        @param K: intrinsic matrix of image
        @param render_pose: Camera to World Matrix, torch.Tensor in shape [3,4] or [4,4]
        @param nerf: the nerf for rendering
        @return: a rgb matrix with shape [H,W,3]
        """
        from datasets.nerf.utils import get_rays_of_a_view
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, render_pose, **self.kwargs)
        ori_shape = rays_o.shape[:-1]
        rays_o, rays_d, viewdirs = rays_o.reshape((-1, 3)), rays_d.reshape((-1, 3)), viewdirs.reshape((-1, 3))
        # limit the batchsize
        rgb = torch.cat([
            nerf(ro.to(nerf.xyz_min), rd.to(nerf.xyz_min), vd.to(nerf.xyz_min))[self.key]
            for ro, rd, vd in zip(rays_o.split(self.bs, 0), rays_d.split(self.bs, 0), viewdirs.split(self.bs, 0))
        ])
        return rgb.reshape(*ori_shape, -1)

    def renderViews(self, HW_list, K_list, render_poses, nerf):
        """
        render views by parameters
        :type render_pose:
        @param HW_list: list of (H,W), where H,W are int.
        @param K_list: list of intrinsic matrix of image, where K is float tensor of [3,3]
        @param render_poses: list of Camera to World Matrix, each is a torch.Tensor in shape of [3,4] or [4,4]
        @param nerf: the nerf for rendering
        @return: a list of rgb matrices with tensor of shape [H_i,W_i,3]
        """
        from datasets.nerf.utils import get_rays_of_a_view
        all_rays_o, all_rays_d, all_dirs, all_ori_shape, all_len = [], [], [], [], []
        for i, (pose, (H, W), K) in enumerate(zip(render_poses, HW_list, K_list)):
            rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, pose, **self.kwargs)
            ori_shape = rays_o.shape[:-1]
            all_rays_o.append(rays_o.reshape((-1, 3)))
            all_rays_d.append(rays_d.reshape((-1, 3)))
            all_dirs.append(viewdirs.reshape((-1, 3)))
            all_ori_shape.append(ori_shape)
            all_len.append(len(rays_o.reshape((-1, 3))))
        all_rays_o, all_rays_d = torch.cat(all_rays_o, dim=0), torch.cat(all_rays_d, dim=0)
        all_dirs = torch.cat(all_dirs, dim=0)
        # limit the batchsize
        ret = torch.cat([
            nerf(ro.to(nerf.xyz_min), rd.to(nerf.xyz_min), vd.to(nerf.xyz_min))[self.key]
            for ro, rd, vd in
            zip(all_rays_o.split(self.bs, 0), all_rays_d.split(self.bs, 0), all_dirs.split(self.bs, 0))
        ])
        ret_list = [rgb.reshape(*ori_shape, -1) for rgb, ori_shape in zip(ret.split(all_len), all_ori_shape)]
        return ret_list
