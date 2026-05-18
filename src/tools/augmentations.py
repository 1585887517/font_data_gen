import cv2
import numpy as np
import random


class Augmentations:

    # ==================================================
    # 🚀 更真实的透视变形
    # ==================================================
    @staticmethod
    def perspective(img, mask):

        h, w = img.shape[:2]

        margin_x = int(w * random.uniform(0.02, 0.15))
        margin_y = int(h * random.uniform(0.02, 0.15))

        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        dst = np.float32([
            [
                random.randint(0, margin_x),
                random.randint(0, margin_y)
            ],
            [
                w - random.randint(0, margin_x),
                random.randint(0, margin_y)
            ],
            [
                w - random.randint(0, margin_x),
                h - random.randint(0, margin_y)
            ],
            [
                random.randint(0, margin_x),
                h - random.randint(0, margin_y)
            ]
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        img = cv2.warpPerspective(
            img,
            M,
            (w, h),
            borderValue=(255, 255, 255)
        )

        mask = cv2.warpPerspective(
            mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        return img, mask


    # ==================================================
    # 🚀 纸张凹凸/局部形变 (Mesh Distortion)
    # ==================================================
    @staticmethod
    def mesh_distortion(img, mask):
        """
        模拟纸张局部凹凸不平的效果
        """
        h, w = img.shape[:2]

        # 网格密度
        grid_steps = random.randint(4, 8)
        grid_h = np.linspace(0, h, grid_steps)
        grid_w = np.linspace(0, w, grid_steps)

        mesh_h, mesh_w = np.meshgrid(grid_h, grid_w, indexing='ij')

        # 给内部网格点增加随机位移
        strength = random.uniform(2.0, 8.5)
        distort_h = mesh_h + np.random.normal(0, strength, mesh_h.shape)
        distort_w = mesh_w + np.random.normal(0, strength, mesh_w.shape)

        # 边界保持不动
        distort_h[0, :] = 0
        distort_h[-1, :] = h
        distort_h[:, 0] = mesh_h[:, 0]
        distort_h[:, -1] = mesh_h[:, -1]

        distort_w[0, :] = mesh_w[0, :]
        distort_w[-1, :] = mesh_w[-1, :]
        distort_w[:, 0] = 0
        distort_w[:, -1] = w

        # 插值生成重映射表
        from scipy.interpolate import RectBivariateSpline
        
        # 简单方案：使用 cv2.resize 插值回全图大小
        map_h = cv2.resize(distort_h, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        map_w = cv2.resize(distort_w, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        img = cv2.remap(img, map_w, map_h, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        mask = cv2.remap(mask, map_w, map_h, interpolation=cv2.INTER_NEAREST, borderValue=0)

        return img, mask


    # ==================================================
    # 🚀 页面弯曲/弧度 (Bending Warp)
    # ==================================================
    @staticmethod
    def bending_warp(img, mask):
        """
        模拟书页弯曲或卷曲的效果
        """
        h, w = img.shape[:2]
        
        # 随机选择弯曲方向（水平或垂直）
        direction = random.choice(['horizontal', 'vertical'])
        
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        
        if direction == 'horizontal':
            # 水平弯曲（圆柱形或正弦形）
            curve = random.uniform(15, 45)
            # 随机相位，模拟不同的弯曲中心
            phase = random.uniform(0, np.pi)
            offset = curve * np.sin(yy / h * np.pi + phase)
            xx = xx + offset
        else:
            # 垂直弯曲
            curve = random.uniform(15, 45)
            phase = random.uniform(0, np.pi)
            offset = curve * np.sin(xx / w * np.pi + phase)
            yy = yy + offset

        img = cv2.remap(img, xx, yy, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        mask = cv2.remap(mask, xx, yy, interpolation=cv2.INTER_NEAREST, borderValue=0)

        return img, mask


    # ==================================================
    # 🚀 小角度扫描旋转
    # ==================================================
    @staticmethod
    def rotate(img, mask):

        angle = random.uniform(-5, 5)

        h, w = img.shape[:2]

        M = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0
        )

        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderValue=(255, 255, 255)
        )

        mask = cv2.warpAffine(
            mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        return img, mask


    # ==================================================
    # 🚀 墨水扩散（非常关键）
    # ==================================================
    @staticmethod
    def ink_bleed(img, mask=None):

        if random.random() < 0.25:

            kernel_size = 2

            kernel = np.ones(
                (kernel_size, kernel_size),
                np.uint8
            )

            dark = 255 - img

            dark = cv2.dilate(
                dark,
                kernel,
                iterations=1
            )

            img = 255 - dark

        if mask is None:
            return img

        return img, mask


    # ==================================================
    # 🚀 扫描模糊
    # ==================================================
    @staticmethod
    def scanner_blur(img, mask=None):

        if random.random() < 0.45:

            k = random.choice([3, 3, 5])

            img = cv2.GaussianBlur(
                img,
                (k, k),
                0
            )

        if mask is None:
            return img

        return img, mask


    @staticmethod
    def expand_label_edges(mask, classes=(1, 2), iterations=1):

        if mask is None:
            return None

        kernel = np.ones((2, 2), np.uint8)
        out = mask.copy()

        for cls in classes:
            cls_region = (mask == cls).astype(np.uint8)
            expanded = cv2.dilate(cls_region, kernel, iterations=iterations) > 0
            out[(out == 0) & expanded] = cls

        return out


    # ==================================================
    # 🚀 JPEG压缩伪影（非常重要）
    # ==================================================
    @staticmethod
    def jpeg_artifact(img):

        if random.random() < 0.25:

            quality = random.randint(65, 92)

            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                quality
            ]

            _, encimg = cv2.imencode(
                '.jpg',
                img,
                encode_param
            )

            img = cv2.imdecode(
                encimg,
                1
            )

        return img


    # ==================================================
    # 🚀 光照不均（扫描件核心）
    # ==================================================
    @staticmethod
    def uneven_illumination(img):

        h, w = img.shape[:2]
        small_h = max(48, h // 8)
        small_w = max(48, w // 8)

        mask = np.zeros((small_h, small_w), dtype=np.float32)

        center_x = random.randint(0, small_w)
        center_y = random.randint(0, small_h)

        radius = random.randint(
            int(min(small_h, small_w) * 0.5),
            int(min(small_h, small_w) * 1.2)
        )

        cv2.circle(
            mask,
            (center_x, center_y),
            radius,
            1,
            -1
        )

        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=random.uniform(18, 34)
        )

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        strength = random.uniform(0.9, 1.08)

        img = img.astype(np.float32)

        for c in range(3):
            img[:, :, c] *= (
                1 + (mask - 0.5) * (strength - 1)
            )

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img


    # ==================================================
    # 🚀 纸张纹理噪声
    # ==================================================
    @staticmethod
    def paper_noise(img):

        noise = np.random.normal(
            0,
            random.uniform(1, 3),
            img.shape[:2]
        )[:, :, None]

        img = img.astype(np.float32)

        img += noise

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)


    # ==================================================
    # 🚀 手机拍摄阴影/曝光/色偏
    # ==================================================
    @staticmethod
    def phone_capture_effects(img):

        img = Augmentations.edge_shadow(img)
        img = Augmentations.local_shadow(img)
        img = Augmentations.color_cast(img)
        img = Augmentations.sensor_noise(img)

        return img


    @staticmethod
    def edge_shadow(img):

        if random.random() > 0.55:
            return img

        h, w = img.shape[:2]
        small_h = max(48, h // 8)
        small_w = max(48, w // 8)
        yy, xx = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
        cx = small_w * random.uniform(0.45, 0.55)
        cy = small_h * random.uniform(0.45, 0.55)

        dist = np.sqrt(((xx - cx) / small_w) ** 2 + ((yy - cy) / small_h) ** 2)
        dist = dist / max(dist.max(), 1e-6)
        strength = random.uniform(0.08, 0.22)
        shade = 1.0 - strength * (dist ** random.uniform(1.2, 2.2))
        shade = cv2.resize(shade, (w, h), interpolation=cv2.INTER_CUBIC)

        out = img.astype(np.float32) * shade[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)


    @staticmethod
    def local_shadow(img):

        if random.random() > 0.35:
            return img

        h, w = img.shape[:2]
        small_h = max(48, h // 8)
        small_w = max(48, w // 8)
        shadow = np.ones((small_h, small_w), dtype=np.float32)

        x1 = random.randint(-small_w // 4, small_w // 2)
        x2 = x1 + random.randint(small_w // 3, int(small_w * 0.9))
        y1 = random.randint(-small_h // 5, small_h)
        y2 = y1 + random.randint(small_h // 4, int(small_h * 0.8))

        cv2.rectangle(
            shadow,
            (x1, y1),
            (x2, y2),
            random.uniform(0.82, 0.94),
            -1
        )

        shadow = cv2.GaussianBlur(
            shadow,
            (0, 0),
            sigmaX=random.uniform(7, 16),
            sigmaY=random.uniform(7, 16)
        )

        shadow = cv2.resize(
            shadow,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        out = img.astype(np.float32) * shadow[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)


    @staticmethod
    def color_cast(img):

        if random.random() > 0.75:
            return img

        gains = np.array(
            [
                random.uniform(0.96, 1.04),
                random.uniform(0.96, 1.04),
                random.uniform(0.93, 1.06)
            ],
            dtype=np.float32
        )

        out = img.astype(np.float32) * gains
        return np.clip(out, 0, 255).astype(np.uint8)


    @staticmethod
    def sensor_noise(img):

        if random.random() > 0.5:
            return img

        noise = np.random.normal(
            0,
            random.uniform(0.8, 2.5),
            img.shape[:2]
        )[:, :, None]

        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)


    # ==================================================
    # 🚀 主入口
    # ==================================================
    @staticmethod
    def scan_noise(img, mask=None, phone_effects=True):

        # 墨水扩散
        if mask is None:
            img = Augmentations.ink_bleed(img)
        else:
            img, mask = Augmentations.ink_bleed(img, mask)

        # 模糊
        if mask is None:
            img = Augmentations.scanner_blur(img)
        else:
            img, mask = Augmentations.scanner_blur(img, mask)

        # 光照不均
        img = Augmentations.uneven_illumination(img)

        # JPEG artifact
        img = Augmentations.jpeg_artifact(img)

        # 纸张噪声
        img = Augmentations.paper_noise(img)

        if phone_effects:
            # 手机拍摄效果
            img = Augmentations.phone_capture_effects(img)

        if mask is None:
            return img

        return img, mask
