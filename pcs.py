import open3d as o3d
import os
import torch

from common.vis import Visualizer as vis

def render(data_path, N, model_name):
    print('data path: ', data_path)
    print('# sample: ', N)
    print('model: ', model_name)

    pcs = os.listdir(data_path)
    # breakpoint()

    for idx in range(0, N):

        n_idx_pcs = []
        for pc in pcs:
            if int(pc[:3]) == idx:
                n_idx_pcs.append(pc)

        task1_pc_paths, task2_pc_paths, task3_pc_paths, task4_pc_paths = [], [] ,[] ,[]
        task5_pc_paths, task6_pc_paths, task7_pc_paths = [], [] ,[]

        for pc in n_idx_pcs:
            if int(pc.split('_')[1]) == 1:
                task1_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 2:
                task2_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 3:
                task3_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 4:
                task4_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 5:
                task5_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 6:
                task6_pc_paths.append(pc)
            elif int(pc.split('_')[1]) == 7:
                task7_pc_paths.append(pc)
        
        
        # # 1: Random R, t
        src_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task1_pc_paths)[0])).points, dtype=torch.float32)
        trg_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task1_pc_paths)[1])).points, dtype=torch.float32)
        cd_cycle = float(sorted(task1_pc_paths)[0].split('_')[5])
        rot_err_cycle = float(sorted(task1_pc_paths)[0].split('_')[6][:-4])
        _1 = [src_pcd, trg_pcd]; savepath='./vis/%03d_1_%s_%3.2f_%.2f.png' % (idx, model_name, cd_cycle, rot_err_cycle)
        if not os.path.exists(savepath): vis.vis_pcd_color(_1, ['lightred', 'lightblue'], savepath=savepath)
        print(f'{idx}/{N}, 1/7 | Rendering done.')

        # # 2: Prediction src2trg
        src_pcd_pr_s2t = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task2_pc_paths)[0])).points, dtype=torch.float32)
        trg_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task2_pc_paths)[1])).points, dtype=torch.float32)
        _2 = [src_pcd_pr_s2t, trg_pcd]; savepath='./vis/%03d_2_%s.png' % (idx, model_name)
        if not os.path.exists(savepath): vis.vis_pcd_color(_2, ['red', 'lightred'], savepath=savepath)
        print(f'{idx}/{N}, 2/7 | Rendering done.')

        # # 3: Prediction trg2src
        src_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task3_pc_paths)[0])).points, dtype=torch.float32)
        trg_pcd_pr_t2s = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task3_pc_paths)[1])).points, dtype=torch.float32)
        _3 = [src_pcd, trg_pcd_pr_t2s]; savepath='./vis/%03d_3_%s.png' % (idx, model_name)
        if not os.path.exists(savepath): vis.vis_pcd_color(_3, ['red', 'lightred'], savepath=savepath)
        print(f'{idx}/{N}, 3/7 | Rendering done.')

        # # 4: Ground-truth src2trg
        src_pcd_gt_s2t = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task4_pc_paths)[0])).points, dtype=torch.float32)
        trg_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task4_pc_paths)[1])).points, dtype=torch.float32)
        _4 = [src_pcd_gt_s2t, trg_pcd]; savepath='./vis/%03d_4_%s.png' % (idx, model_name)
        if not os.path.exists(savepath): vis.vis_pcd_color(_4, ['blue', 'lightblue'], savepath=savepath)
        print(f'{idx}/{N}, 4/7 | Rendering done.')

        # # 5: Ground-truth trg2src
        src_pcd = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task5_pc_paths)[0])).points, dtype=torch.float32)
        trg_pcd_gt_t2s = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task5_pc_paths)[1])).points, dtype=torch.float32)
        _5 = [src_pcd, trg_pcd_gt_t2s]; savepath='./vis/%03d_5_%s.png' % (idx, model_name)
        if not os.path.exists(savepath): vis.vis_pcd_color(_5, ['blue', 'lightblue'], savepath=savepath)
        print(f'{idx}/{N}, 5/7 | Rendering done.')

        # # 6: Assembly results comparison: src2trg (2+4)
        gt_assembly = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task6_pc_paths)[0])).points, dtype=torch.float32)
        pred_assembly = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task6_pc_paths)[1])).points, dtype=torch.float32)
        cd_s2t = float(sorted(task6_pc_paths)[0].split('_')[5])
        rot_err_s2t = float(sorted(task6_pc_paths)[0].split('_')[6][:-4])
        _6 = [gt_assembly, pred_assembly]; savepath='./vis/%03d_6_%s_%3.2f_%.2f.png' % (idx, model_name, cd_s2t, rot_err_s2t)
        if not os.path.exists(savepath): vis.vis_pcd_color(_6, ['darkred', 'blue'], savepath=savepath)
        print(f'{idx}/{N}, 6/7 | Rendering done.')

        # # 7: Assembly results comparison: trg2src (3+5)
        gt_assembly = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task7_pc_paths)[0])).points, dtype=torch.float32)
        pred_assembly = torch.tensor(o3d.io.read_point_cloud(os.path.join(data_path, sorted(task7_pc_paths)[1])).points, dtype=torch.float32)
        cd_t2s = float(sorted(task7_pc_paths)[0].split('_')[5])
        rot_err_t2s = float(sorted(task7_pc_paths)[0].split('_')[6][:-4])
        _7 = [gt_assembly, pred_assembly]; savepath='./vis/%03d_7_%s_%3.2f_%.2f.png' % (idx, model_name, cd_t2s, rot_err_t2s)
        if not os.path.exists(savepath): vis.vis_pcd_color(_7, ['darkred', 'blue'], savepath=savepath)
        print(f'{idx}/{N}, 7/7 | Rendering done.')

    return

if __name__ == '__main__':

    vis.initialize()
    
    data_path = '../scp/scp'
    N = 403
    model_name = '_global'

    render(data_path, N, model_name)