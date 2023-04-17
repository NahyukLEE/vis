r""" Visualize model predictions """
import os
import time

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import open3d as o3d
import torch.nn.functional as F

from . import utils


class Visualizer:

    @classmethod
    def vis_trimesh(cls, mesh_t, pcds=None, pcds_normals=None, v_normal=False, t_normal=False):
        colors = list(cls.colors.keys())

        if pcds is not None:
            pcds_o = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())) for pcd in pcds]
        else:
            pcds_o = []

        if pcds is not None and pcds_normals is not None:
            pcds_normals_o = []
            for p, n in zip(pcds, pcds_normals):
                l = torch.arange(len(p))
                l = torch.stack([l, l + len(p)]).t().contiguous()
                l_o = o3d.geometry.LineSet()
                l_o.points = o3d.utility.Vector3dVector(torch.cat([p, p + n * 0.05]))
                l_o.lines = o3d.utility.Vector2iVector(l)
                pcds_normals_o.append(l_o)
        else:
            pcds_normals_o = []

        mesh_o = []
        for idx, t in enumerate(mesh_t):
            o = o3d.geometry.TriangleMesh()
            o.vertices = o3d.utility.Vector3dVector(t.vertices)
            o.triangles = o3d.utility.Vector3iVector(t.faces)
            o.compute_vertex_normals()
            o.paint_uniform_color(cls.colors[colors[idx]])
            o.compute_triangle_normals()
            o.compute_vertex_normals()

            # Vertex normal visualization
            if v_normal:
                v = torch.tensor(o.vertices)
                l = torch.arange(len(v))
                l = torch.stack([l, l + len(v)]).t().contiguous()
                l_o = o3d.geometry.LineSet()
                normal = F.normalize(torch.tensor(o.vertex_normals), dim=-1)
                l_o.points = o3d.utility.Vector3dVector(torch.cat([v, normal]))
                l_o.lines = o3d.utility.Vector2iVector(l)
                v_o = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
                mesh_o.append(l_o)
                mesh_o.append(v_o)

            # Triangle normal visualization
            if t_normal:
                v = torch.tensor(o.vertices)
                t = torch.tensor(o.triangles)
                tc = v[t.view(-1).contiguous().long(), :].view(t.size(0), 3, 3).contiguous().mean(dim=1)
                tc_o = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tc))
                normal = F.normalize(torch.tensor(o.triangle_normals), dim=-1)
                l = torch.arange(len(tc))
                l = torch.stack([l, l + len(tc)]).t().contiguous()
                l_o = o3d.geometry.LineSet()
                l_o.points = o3d.utility.Vector3dVector(torch.cat([tc, tc+normal]))
                l_o.lines = o3d.utility.Vector2iVector(l)
                mesh_o.append(l_o)
                mesh_o.append(tc_o)

            mesh_o.append(o)
        o3d.visualization.draw_geometries(mesh_o + pcds_o + pcds_normals_o)


    @classmethod
    def vis_pcd(cls, pcds):
        pcds_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())) for pcd in pcds]
        o3d.visualization.draw_geometries(pcds_o3d)

    @classmethod
    def vis_pcd_color(cls, pcds, colors, savepath=None):

        pcds_o3d = []
        for idx, (pcd, color) in enumerate(zip(pcds, colors)):
            pcds_o3d.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())).paint_uniform_color(cls.colors[color]))

        if savepath:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for p in pcds_o3d:
                vis.add_geometry(p)
                vis.update_geometry(p)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.2)
            vis.capture_screen_image(savepath)
            time.sleep(0.2)
            vis.destroy_window()
        else:
            o3d.visualization.draw_geometries(pcds_o3d)

    @classmethod
    def vis_pcd_rgb(cls, pcd1, pcd2=None, pcd3=None, pcd4=None, savepath=None, whitelast=False):
        pcd1_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())).paint_uniform_color(cls.colors['red']) for pcd in pcd1]
        pcd2_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())).paint_uniform_color(cls.colors['blue']) for pcd in pcd2] if pcd2 is not None else []
        pcd3_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())).paint_uniform_color(cls.colors['green']) for pcd in pcd3] if pcd3 is not None else []
        pcd4_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())).paint_uniform_color(cls.colors['orange']) for pcd in pcd4] if pcd4 is not None else []
        pcd = pcd1_o3d + pcd2_o3d + pcd3_o3d + pcd4_o3d
        if savepath:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for p in pcd:
                vis.add_geometry(p)
                vis.update_geometry(p)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(savepath)
            vis.destroy_window()
        else:
            o3d.visualization.draw_geometries(pcd)

    @classmethod
    def vis_pcd_nd(cls, pcds, nds):
        pcds_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu())) for pcd in pcds]
        o3d.visualization.draw_geometries(pcds_o3d)

        nds_o3d = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nd.cpu())) for nd in nds]
        o3d.visualization.draw_geometries(nds_o3d)

    @classmethod
    def visualize_trimesh_mating_surf(cls, mesh_t, surface_pts_flatten, mating_ids, transformation=None, trans_scale=0.5):
        mids = [[(int(y.split('-')[0]), int(y.split('-')[1])) for y in x.split(':')] for x in mating_ids.split('_')]
        mating_objs, mating_ncorr = [o[0] for o in mids], [o[1] for o in mids]
        sidx = 0
        surface_pts = {}
        for mo, mc in zip(mating_objs, mating_ncorr):
            key = '%d-%d' % (mo[0], mo[1])
            value = (surface_pts_flatten[sidx:sidx + mc[0]], surface_pts_flatten[sidx + mc[0]:sidx + mc[0] + mc[1]])
            sidx += mc[0] + mc[1]
            surface_pts[key] = value
        colors = list(cls.colors.keys())

        if transformation is not None:
            gt_trans, gt_rotat = transformation
        else:
            rotat = torch.zeros((3, 3))
            for i in range(3): rotat[i, i] = 1.
            gt_rotat = rotat.unsqueeze(0).repeat(len(mesh_t), 1, 1)
            gt_trans = torch.zeros(len(mesh_t), 3)

        # Translation for better view
        obj_centers = []
        centroid = 0
        n_vertices = 0
        for mesh in mesh_t:
            vertices = torch.tensor(mesh.vertices).float()
            obj_centers.append(vertices.mean(dim=0))
            centroid += vertices.sum(dim=0)
            n_vertices += vertices.size(0)
        centroid /= n_vertices
        obj_centers = torch.stack(obj_centers)

        trans_vec = []
        for oc in obj_centers:
            trans_vec.append((centroid - oc) * trans_scale)
        trans_vec = torch.stack(trans_vec)

        mesh_o, mesh_color = [], []
        for idx, (mesh, trans, gtt, gtr) in enumerate(zip(mesh_t, trans_vec, gt_trans, gt_rotat)):
            mesh_color.append(cls.colors[colors[idx]])
            o = o3d.geometry.TriangleMesh()
            # trans_vert = (gtr.inverse() @ (torch.tensor(mesh.vertices).float() - trans + gtt).t()).t().numpy()
            trans_vert = ((gtr @ (torch.tensor(mesh.vertices).float() - gtt).t()).t() - trans).numpy()
            # trans_vert = ((gtr @ (torch.tensor(mesh.vertices).float() - gtt).t()).t()).numpy()
            o.vertices = o3d.utility.Vector3dVector(trans_vert)
            o.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o.compute_vertex_normals()
            o.paint_uniform_color(mesh_color[idx])
            mesh_o.append(o)

        for m in surface_pts:
            obj1, obj2 = [int(x) for x in m.split('-')]
            darken_amt = 0.2
            color1 = tuple([max(0.0, x - darken_amt) for x in mesh_color[obj1]])
            color2 = tuple([max(0.0, x - darken_amt) for x in mesh_color[obj2]])
            surface_pts1 = surface_pts[m][0]
            surface_pts2 = surface_pts[m][1]

            if surface_pts1.size(0) == 0 or surface_pts2.size(0) == 0:
                print()

            gtt1, gtt2 = gt_trans[obj1], gt_trans[obj2]
            gtr1, gtr2 = gt_rotat[obj1], gt_rotat[obj2]

            # sf_pts1_1 = surface_pts1 - trans_vec[obj1]
            # sf_pts1_2 = (gtr2 @ ((gtr1.inverse() @ surface_pts1.t()).t().contiguous() + gtt1 - gtt2).t()).t().contiguous() - trans_vec[obj2]
            sf_pts1_1 = surface_pts1 - trans_vec[obj2]
            sf_pts1_2 = (gtr1 @ ((gtr2.inverse() @ surface_pts1.t()).t().contiguous() + gtt2 - gtt1).t()).t().contiguous() - trans_vec[obj1]
            pcd1_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sf_pts1_1))
            pcd1_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sf_pts1_2))
            pcd1_1.paint_uniform_color(color1)
            pcd1_2.paint_uniform_color(color1)

            lines1 = torch.arange(sf_pts1_1.size(0))
            lines1 = torch.stack([lines1, lines1 + sf_pts1_2.size(0)]).t().contiguous()
            points1 = torch.cat([sf_pts1_1, sf_pts1_2])
            lineset1 = o3d.geometry.LineSet()
            lineset1.points = o3d.utility.Vector3dVector(points1)
            lineset1.lines = o3d.utility.Vector2iVector(lines1)
            lineset1.paint_uniform_color(color1)

            # sf_pts2_1 = surface_pts2 - trans_vec[obj2]
            # sf_pts2_2 = (gtr1 @ ((gtr2.inverse() @ surface_pts2.t()).t().contiguous() + gtt2 - gtt1).t()).t().contiguous() - trans_vec[obj1]
            sf_pts2_1 = surface_pts2 - trans_vec[obj1]
            sf_pts2_2 = (gtr2 @ ((gtr1.inverse() @ surface_pts2.t()).t().contiguous() + gtt1 - gtt2).t()).t().contiguous() - trans_vec[obj2]
            pcd2_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sf_pts2_1))
            pcd2_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sf_pts2_2))
            pcd2_1.paint_uniform_color(color2)
            pcd2_2.paint_uniform_color(color2)

            lines2 = torch.arange(sf_pts2_1.size(0))
            lines2 = torch.stack([lines2, lines2 + sf_pts2_2.size(0)]).t().contiguous()
            points2 = torch.cat([sf_pts2_1, sf_pts2_2])
            lineset2 = o3d.geometry.LineSet()
            lineset2.points = o3d.utility.Vector3dVector(points2)
            lineset2.lines = o3d.utility.Vector2iVector(lines2)
            lineset2.paint_uniform_color(color2)

            mesh_o += [pcd1_1, pcd2_1, lineset1, lineset2]

        o3d.visualization.draw_geometries(mesh_o)

    @classmethod
    def initialize(cls):
        cls.colors = {'red': [255, 50, 50],
                      'blue': [102, 140, 255],
                      'green': [80, 194, 62],
                      'grey': [191, 191, 191],
                      'cyan': [134, 230, 209],
                      'pink': [239, 166, 255],
                      'purple': [177, 64, 202],
                      'yellow': [245, 237, 77],
                      'orange': [245, 159, 77],
                      'black': [100, 100, 100],
                      'lightblue': [181, 183, 240],
                      'darkgreen': [87, 138, 91],
                      'forest': [0, 205, 16],
                      'darkred': [153, 56, 56],
                      'darkcyan': [47, 190, 151],
                      'lightgrey': [228, 228, 228],
                      'darkorange': [214, 97, 0],
                      'lightred': [255, 191, 191],
                      'white': [255, 255, 255],
                      'strongpurple': [98, 0, 240]
                      }
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.vis_path = './vis/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)