import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import random


class DocumentGenerator:

    def __init__(self, cfg):

        self.cfg = cfg

        self.font_paths = self._discover_fonts()
        self.font_sizes = [18, 22, 26, 30, 34, 38, 42, 48]
        self.font_cache = {}
        for font_path in self.font_paths:
            for size in self.font_sizes:
                self.font_cache[(font_path, size)] = ImageFont.truetype(font_path, size)

        self.texts = [
            "Invoice 12345",
            "AI Document",
            "中文测试文档",
            "OCR segmentation",
            "合同编号 A-001",
            "CONFIDENTIAL",
            "Meeting Notes",
            "扫描文档测试",
            "Printed Text Example",
            "客户名称：上海某某科技有限公司",
            "Amount Due: 12,345.67",
            "日期：2026-05-09",
            "Please review and approve",
            "收货地址：北京市朝阳区示例路88号",
            "PO No. 2026-0511",
            "Subtotal: 893.20",
            "Tax ID: 91310000MA1K0000X",
            "Project: Document Capture",
            "规格型号：A4-STD",
            "开户银行：中国工商银行",
            "经办人：王晓明",
            "Total pages: 3",
            "Original / Copy",
            "This document is machine generated"
        ]


    def _discover_fonts(self):

        font_paths = []

        if os.path.isdir(self.cfg.FONT_ROOT):
            for name in sorted(os.listdir(self.cfg.FONT_ROOT)):
                if name.lower().endswith((".otf", ".ttf", ".ttc")):
                    font_paths.append(os.path.join(self.cfg.FONT_ROOT, name))

        if not font_paths:
            font_paths.append(self.cfg.FONT_PATH)

        return font_paths


    def _font(self, size):

        nearest_size = min(self.font_sizes, key=lambda s: abs(s - size))
        font_path = random.choice(self.font_paths)
        return self.font_cache[(font_path, nearest_size)]


    # ==================================================
    # 🚀 主入口
    # ==================================================
    def build(self):

        # ==================================================
        # 1. 创建纸张背景
        # ==================================================
        if self.cfg.ENABLE_PAPER_TEXTURE:
            img = Image.fromarray(self._create_paper_background())
        else:
            img = Image.new(
                "RGB",
                (self.cfg.WIDTH, self.cfg.HEIGHT),
                (255, 255, 255)
            )

        draw = ImageDraw.Draw(img)

        # ==================================================
        # 2. segmentation mask
        # 0 background
        # 1 printed
        # ==================================================
        mask = np.zeros(
            (self.cfg.HEIGHT, self.cfg.WIDTH),
            dtype=np.uint8
        )

        prob_total = (
            self.cfg.FORM_LAYOUT_PROB
            + self.cfg.RECEIPT_LAYOUT_PROB
            + self.cfg.FREE_LAYOUT_PROB
        )
        layout_roll = random.random() * prob_total
        form_prob = self.cfg.FORM_LAYOUT_PROB
        receipt_prob = self.cfg.RECEIPT_LAYOUT_PROB

        if self.cfg.ENABLE_FORM_LAYOUT and layout_roll < form_prob:
            self._draw_form_layout(draw)
            self._draw_structured_printed_text(draw, mask)
        elif layout_roll < form_prob + receipt_prob:
            self._draw_receipt_layout(draw, mask)
        else:
            self._draw_free_printed_text(draw, mask)

        # ==================================================
        # 🚀 转numpy
        # ==================================================
        img = np.array(img)

        # ==================================================
        # 🚀 模拟打印退化（核心）
        # ==================================================
        img, mask = self._simulate_printing(img, mask)

        return img, mask


    def _draw_free_printed_text(self, draw, mask):

        y = random.randint(22, 76)

        line_count = random.randint(10, 28)
        columns = random.choice([1, 1, 2])
        column_x = [random.randint(24, 80)]
        if columns == 2:
            column_x.append(random.randint(self.cfg.WIDTH // 2, self.cfg.WIDTH // 2 + 60))

        for _ in range(line_count):

            text = random.choice(self.texts)
            font = self._font(random.choice([22, 26, 30, 34, 38]))

            x = random.choice(column_x) + random.randint(0, 36)

            # ==================================================
            # 🚀 模拟打印深浅（关键）
            # ==================================================
            ink = random.randint(0, 105)

            color = (ink, ink, ink)

            self._draw_printed_text(draw, mask, text, (x, y), font, color)

            # ==================================================
            # 🚀 行间距随机
            # ==================================================
            bbox = font.getbbox(text)

            text_h = bbox[3] - bbox[1]

            y += text_h + random.randint(6, 30)

            if y > self.cfg.HEIGHT - 70 and columns == 2 and len(column_x) == 2:
                y = random.randint(28, 70)
                column_x.pop(0)
            elif y > self.cfg.HEIGHT - 60:
                break


    def _draw_structured_printed_text(self, draw, mask):

        title_font = self._font(random.choice([38, 42, 48]))
        label_font = self._font(random.choice([22, 26, 30]))
        body_font = self._font(random.choice([22, 26, 30]))

        title = random.choice([
            "费用报销单",
            "收货确认单",
            "Document Review Form",
            "合同审批表",
            "Inspection Report"
        ])

        self._draw_printed_text(
            draw,
            mask,
            title,
            (random.randint(310, 380), random.randint(24, 42)),
            title_font,
            (random.randint(0, 45),) * 3
        )

        fields = [
            "客户名称", "合同编号", "联系人", "联系电话",
            "日期", "金额", "地址", "备注",
            "Item", "Qty", "Unit Price", "Approved By"
        ]

        x0 = random.randint(44, 84)
        y0 = random.randint(104, 136)
        row_h = random.randint(42, 64)
        col_w = random.choice([
            [130, 330, 150, 250],
            [165, 280, 145, 270],
            [120, 250, 170, 320],
        ])
        rows = random.randint(5, 11)

        for r in range(rows):
            y = y0 + r * row_h + random.randint(7, 13)
            for c in range(0, 4, 2):
                x = x0 + sum(col_w[:c]) + 12
                text = random.choice(fields) + ":"
                self._draw_printed_text(
                    draw,
                    mask,
                    text,
                    (x, y),
                    label_font,
                    (random.randint(15, 85),) * 3
                )

        paragraph_y = y0 + rows * row_h + random.randint(22, 34)
        for i in range(random.randint(3, 6)):
            text = random.choice(self.texts)
            self._draw_printed_text(
                draw,
                mask,
                text,
                (random.randint(75, 130), paragraph_y + i * random.randint(34, 42)),
                body_font,
                (random.randint(20, 90),) * 3
            )


    def _draw_receipt_layout(self, draw, mask):

        title_font = self._font(random.choice([30, 34, 38]))
        item_font = self._font(random.choice([18, 22, 26]))
        total_font = self._font(random.choice([26, 30, 34]))

        x0 = random.randint(145, 230)
        width = random.randint(450, 650)
        y = random.randint(30, 70)
        right = min(self.cfg.WIDTH - 30, x0 + width)

        self._draw_printed_text(
            draw,
            mask,
            random.choice(["SALES INVOICE", "费用明细", "Delivery Note", "收款凭证"]),
            (x0 + random.randint(60, 150), y),
            title_font,
            (random.randint(0, 65),) * 3
        )

        y += random.randint(42, 58)
        for _ in range(random.randint(12, 24)):
            left = random.choice(self.texts)
            amount = random.choice(["12.00", "35.80", "108.50", "1,245.00", "Qty 2", "OK"])
            left_len = random.randint(min(4, len(left)), min(len(left), 28))
            self._draw_printed_text(
                draw,
                mask,
                left[:left_len],
                (x0 + random.randint(0, 24), y),
                item_font,
                (random.randint(10, 100),) * 3
            )
            self._draw_printed_text(
                draw,
                mask,
                amount,
                (right - random.randint(95, 145), y + random.randint(-2, 2)),
                item_font,
                (random.randint(10, 100),) * 3
            )
            if random.random() < 0.3:
                draw.line([x0, y + 28, right, y + 28], fill=(random.randint(180, 220),) * 3, width=1)
            y += random.randint(26, 38)
            if y > self.cfg.HEIGHT - 95:
                break

        if random.random() < 0.85:
            draw.line([x0, y + 4, right, y + 4], fill=(random.randint(90, 160),) * 3, width=random.choice([1, 2]))
            self._draw_printed_text(
                draw,
                mask,
                random.choice(["TOTAL", "合计", "Amount Due"]),
                (x0 + random.randint(4, 30), y + 14),
                total_font,
                (random.randint(0, 65),) * 3
            )
            self._draw_printed_text(
                draw,
                mask,
                random.choice(["128.50", "1,245.00", "12,345.67"]),
                (right - random.randint(130, 180), y + 14),
                total_font,
                (random.randint(0, 65),) * 3
            )


    def _draw_printed_text(self, draw, mask, text, xy, font, color):

        draw.text(
            xy,
            text,
            fill=color,
            font=font
        )

        bbox = draw.textbbox(xy, text, font=font)
        pad = 3
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(self.cfg.WIDTH, bbox[2] + pad)
        y1 = min(self.cfg.HEIGHT, bbox[3] + pad)

        if x1 <= x0 or y1 <= y0:
            return

        text_mask_img = Image.new(
            "L",
            (x1 - x0, y1 - y0),
            0
        )

        text_draw = ImageDraw.Draw(text_mask_img)

        text_draw.text(
            (xy[0] - x0, xy[1] - y0),
            text,
            fill=255,
            font=font
        )

        text_mask = np.array(text_mask_img)
        blur_mask = cv2.GaussianBlur(text_mask, (3, 3), 0)
        mask[y0:y1, x0:x1][blur_mask > 32] = 1


    def _draw_form_layout(self, draw):

        W = self.cfg.WIDTH
        H = self.cfg.HEIGHT

        margin = random.randint(42, 72)
        top = random.randint(92, 122)
        bottom = H - random.randint(70, 110)
        right = W - margin

        line_color = random.choice([
            (145, 145, 145),
            (165, 165, 165),
            (185, 185, 185),
            (115, 130, 150)
        ])

        width = random.choice([1, 1, 2])

        draw.rectangle(
            [margin, top, right, bottom],
            outline=line_color,
            width=width
        )

        row_h = random.randint(42, 64)
        y = top + row_h

        while y < bottom - row_h:
            draw.line([margin, y, right, y], fill=line_color, width=width)
            y += row_h

        col_count = random.choice([2, 3, 4])
        col_positions = sorted({
            margin + random.randint(115, max(120, right - margin - 80))
            for _ in range(col_count)
        })

        for x in col_positions:
            if margin < x < right:
                draw.line([x, top, x, bottom], fill=line_color, width=width)

        if random.random() < 0.55:
            for _ in range(random.randint(2, 8)):
                cx = random.randint(margin + 16, right - 16)
                cy = random.randint(top + 16, bottom - 16)
                draw.rectangle([cx, cy, cx + 12, cy + 12], outline=line_color, width=1)

        if random.random() < 0.45:
            stamp_color = random.choice([(170, 35, 35), (190, 55, 55)])
            cx = random.randint(W - 230, W - 130)
            cy = random.randint(70, 145)
            draw.ellipse(
                [cx - 44, cy - 44, cx + 44, cy + 44],
                outline=stamp_color,
                width=2
            )


    def _create_paper_background(self):

        H = self.cfg.HEIGHT
        W = self.cfg.WIDTH

        base = random.randint(232, 253)
        paper = np.full((H, W, 3), base, dtype=np.float32)

        small_h = max(32, H // 8)
        small_w = max(32, W // 8)
        coarse_small = np.random.normal(
            0,
            random.uniform(2.0, 8.5),
            (small_h, small_w)
        ).astype(np.float32)
        coarse = cv2.resize(
            coarse_small,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )[:, :, None]
        fine = np.random.normal(0, random.uniform(0.8, 3.0), (H, W, 1))

        y_grad = np.linspace(
            random.uniform(-5, 2),
            random.uniform(0, 8),
            H,
            dtype=np.float32
        )[:, None, None]
        x_grad = np.linspace(
            random.uniform(-4, 4),
            random.uniform(-4, 4),
            W,
            dtype=np.float32
        )[None, :, None]

        tint = np.array(
            [
                random.uniform(0.98, 1.03),
                random.uniform(0.98, 1.03),
                random.uniform(0.96, 1.02)
            ],
            dtype=np.float32
        )

        paper = (paper + coarse + fine + y_grad + x_grad) * tint

        if random.random() < 0.35:
            fold_x = random.randint(W // 5, W - W // 5)
            fold = np.exp(-((np.arange(W) - fold_x) ** 2) / (2 * random.uniform(6, 18) ** 2))
            fold = fold[None, :, None] * random.uniform(-10, 8)
            paper += fold

        if random.random() < 0.30:
            yy, xx = np.mgrid[0:H, 0:W]
            cx = random.randint(-W // 4, W + W // 4)
            cy = random.randint(-H // 4, H + H // 4)
            dist = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
            vignette = 1.0 - random.uniform(0.04, 0.16) * np.clip(dist, 0, 1.2)
            paper *= vignette[:, :, None]

        return np.clip(paper, 0, 255).astype(np.uint8)


    # ==================================================
    # 🚀 打印机/扫描件退化
    # ==================================================
    def _simulate_printing(self, img, mask):

        # ==================================================
        # 1. 轻微模糊
        # ==================================================
        if random.random() < 0.45:

            k = random.choice([3, 3, 5])

            img = cv2.GaussianBlur(
                img,
                (k, k),
                0
            )
            mask = self._expand_mask_classes(mask, classes=(1,), iterations=1)

        # ==================================================
        # 2. 墨水扩散
        # ==================================================
        if random.random() < 0.25:

            dark = 255 - img
            kernel = np.ones((2, 2), np.uint8)

            dark = cv2.dilate(
                dark,
                kernel,
                iterations=1
            )

            img = 255 - dark
            mask = self._expand_mask_classes(mask, classes=(1,), iterations=1)

        # ==================================================
        # 3. JPEG artifact
        # ==================================================
        if random.random() < 0.12:

            quality = random.randint(72, 92)

            _, enc = cv2.imencode(
                ".jpg",
                img,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    quality
                ]
            )

            img = cv2.imdecode(enc, 1)

        # ==================================================
        # 4. 扫描噪声
        # ==================================================
        noise = np.random.normal(
            0,
            random.uniform(1, 4),
            img.shape[:2]
        )[:, :, None]

        img = img.astype(np.float32)

        img += noise

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8), mask


    @staticmethod
    def _expand_mask_classes(mask, classes, iterations=1):

        kernel = np.ones((2, 2), np.uint8)
        out = mask.copy()

        for cls in classes:
            cls_region = (mask == cls).astype(np.uint8)
            expanded = cv2.dilate(cls_region, kernel, iterations=iterations) > 0
            out[(out == 0) & expanded] = cls

        return out
