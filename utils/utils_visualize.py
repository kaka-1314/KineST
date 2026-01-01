# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os

import cv2
import numpy as np
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer

from body_visualizer.tools.vis_tools import colors

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"


class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white) / 255.0
        self.black = np.array(black) / 255.0
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=50.0, ylength=50.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: trimesh.Trimesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)

        # now compose as mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(verts) + np.array([-5, -5, 0]), faces=np.array(faces), process=False, face_colors=np.array(texts))

        return mesh


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""

##这个函数以body_pose作为主要的输入，其他的参数主要为定义一些路径，建议自己打断点过一边会比较清楚
# def save_animation(body_pose, savepath, bm, image_path,save_filename,fps=60, resolution=(800, 800),):
def save_animation(body_pose, savepath, bm, fps=60, resolution=(800, 800), ):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    ##从body_model中获得mesh中需要的faces参数
    faces = c2c(bm.f)
    img_array = []
    ##对于body_pose的vertices的每一个frame_id，建立该帧对应的mesh
    for fId in tqdm(range(body_pose.v.shape[0])):
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose.v[fId]),
            faces=faces,
            vertex_colors=np.tile(colors["purple"], (6890, 1)),
        )
        ##可视化基准的棋盘格
        generator = CheckerBoard()
        checker_mesh = generator.gen_checker_xy(generator.black, generator.white)
        ##对生成的mesh进行平移、旋转、缩放等变换
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        ##建立静态场景，并渲染成图片，把图片保存在数组中，最后合成视频
        ##！！！！！需要注意的是，如果要使用trimesh.show打开窗口显示3d的mesh文件的话，需要注释掉mv（第80行）和下面的相关内容
        ##否则trimesh的窗口无法打开
        mv.set_static_meshes([checker_mesh, body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        # if fId %30 ==0:
        #     path = os.path.join(image_path,save_filename+'_{}.jpg'.format(fId))
        #     cv2.imwrite(path,body_image)
        img_array.append(body_image)
    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"DIVX"), fps, resolution)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    ##以下是基于上述脚本改动的一些小脚本，自己看看就能明白
# def save_animation_compare(body_pose, savepath, bm, image_path,save_filename,fps=60, resolution=(800, 800),):
def save_animation_compare(body_pose, savepath, bm, fps=60, resolution=(800, 800), ):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in tqdm(range(body_pose[0].v.shape[0])):
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose[0].v[fId]),
            faces=faces,
            vertex_colors=np.tile(colors["purple"], (6890, 1)),
        )
        body_mesh_pure = trimesh.Trimesh(
            vertices=c2c(body_pose[1].v[fId]),
            faces=faces,
            vertex_colors=np.tile(colors["green"], (6890, 1)),
        )
        body_mesh_gt = trimesh.Trimesh(
            vertices=c2c(body_pose[2].v[fId]),
            faces=faces,
            vertex_colors=np.tile(colors["brown"], (6890, 1)),
        )
        generator = CheckerBoard()
        checker_mesh = generator.gen_checker_xy(generator.black, generator.white)
        ###transform_pose1
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        body_mesh.apply_translation((0,0,1))
        ##transform_pose2
        body_mesh_pure.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh_pure.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh_pure.apply_transform(trimesh.transformations.scale_matrix(0.5))
        body_mesh_pure.apply_translation((0,0,-1))
        ###transform_pose3
        body_mesh_gt.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh_gt.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh_gt.apply_transform(trimesh.transformations.scale_matrix(0.5))

        ##transform_done
        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        mv.set_static_meshes([checker_mesh, body_mesh, body_mesh_gt,body_mesh_pure])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        # if fId %30 ==0:
        #     path = os.path.join(image_path,save_filename+'_{}.jpg'.format(fId))
        #     cv2.imwrite(path,body_image)
        img_array.append(body_image)
    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"DIVX"), fps, resolution)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def save_animation_new(body_pose, bm,resolution=(800, 800),):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in tqdm(range(body_pose.v.shape[0])):
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose.v[fId]),
            faces=faces,
            vertex_colors=np.tile(colors["purple"], (6890, 1)),
        )
        generator = CheckerBoard()
        checker_mesh = generator.gen_checker_xy(generator.black, generator.white)

        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        checker_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        #body_mesh.show()
        print('a')
        mv.set_static_meshes([checker_mesh, body_mesh])
        #
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('a',body_image)
        cv2.waitKey(0)
        print('a')
