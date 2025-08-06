import numpy as np
import cv2
from PIL import Image
import random
import os


class Watermark:
    def __init__(self, secret_key=None, watermark_size=(8, 8)):
        """初始化水印处理器

        Args:
            secret_key: 用于生成伪随机序列的密钥
            watermark_size: 水印大小，默认为8x8
        """
        self.watermark_size = watermark_size
        self.block_size = 8  # DCT块大小
        self.alpha = 0.05  # 水印强度因子，值越大水印越明显但鲁棒性越好

        # 设置随机种子，确保嵌入和提取时使用相同的随机序列
        if secret_key is None:
            secret_key = 12345  # 默认密钥
        self.secret_key = secret_key
        random.seed(secret_key)

        # 生成随机位置序列，用于选择嵌入水印的DCT块
        self.positions = self._generate_positions()

    def _generate_positions(self):
        """生成用于嵌入水印的随机位置序列"""
        positions = []
        # 为8x8的水印生成64个随机位置
        for _ in range(np.prod(self.watermark_size)):
            # 生成随机的块坐标，范围可以根据常见图片尺寸调整
            x = random.randint(0, 30)
            y = random.randint(0, 30)
            positions.append((x, y))
        return positions

    def generate_watermark(self, message=None):
        """生成二进制水印图案

        Args:
            message: 可选的二进制消息，如果未提供则生成随机水印

        Returns:
            二进制水印图案（ndarray）
        """
        if message is not None:
            # 从消息生成水印
            msg_array = np.array(list(message), dtype=np.uint8)
            watermark = msg_array.reshape(self.watermark_size)
        else:
            # 生成随机二进制水印
            watermark = np.random.randint(0, 2, self.watermark_size, dtype=np.uint8)

        return watermark

    def embed(self, image_path, watermark, output_path=None):
        """在图像中嵌入水印

        Args:
            image_path: 原始图像路径
            watermark: 要嵌入的水印图案
            output_path: 嵌入水印后的图像保存路径

        Returns:
            嵌入水印后的图像（ndarray）
        """
        # 读取图像并转换为YCrCb颜色空间，我们只在Y通道（亮度）嵌入水印
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)

        # 将水印展平以便处理
        watermark_flat = watermark.flatten()

        # 对图像进行分块DCT变换并嵌入水印
        for i, (x, y) in enumerate(self.positions):
            # 提取8x8块
            block = y_channel[x * self.block_size:(x + 1) * self.block_size,
                    y * self.block_size:(y + 1) * self.block_size]

            # 进行DCT变换
            dct_block = cv2.dct(block)

            # 在DCT域嵌入水印，选择中频系数（对视觉影响小且鲁棒性较好）
            # 将二进制水印值(-1, 1)映射
            watermark_bit = 2 * watermark_flat[i] - 1
            dct_block[4, 4] += self.alpha * watermark_bit  # 选择(4,4)位置的系数

            # 逆DCT变换
            idct_block = cv2.idct(dct_block)
            y_channel[x * self.block_size:(x + 1) * self.block_size,
            y * self.block_size:(y + 1) * self.block_size] = idct_block

        # 更新Y通道并转换回BGR颜色空间
        ycrcb[:, :, 0] = y_channel.astype(np.uint8)
        watermarked_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # 保存图像（如果指定了输出路径）
        if output_path:
            cv2.imwrite(output_path, watermarked_image)
            print(f"已保存带水印的图像至: {output_path}")

        return watermarked_image

    def extract(self, watermarked_image_path):
        """从图像中提取水印

        Args:
            watermarked_image_path: 带水印的图像路径

        Returns:
            提取的水印图案（ndarray）
        """
        # 读取带水印的图像并转换为YCrCb颜色空间
        watermarked_image = cv2.imread(watermarked_image_path)
        if watermarked_image is None:
            raise ValueError(f"无法读取图像: {watermarked_image_path}")

        ycrcb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)

        # 初始化提取的水印
        extracted_watermark = np.zeros(np.prod(self.watermark_size), dtype=np.uint8)

        # 提取水印
        for i, (x, y) in enumerate(self.positions):
            # 提取8x8块
            block = y_channel[x * self.block_size:(x + 1) * self.block_size,
                    y * self.block_size:(y + 1) * self.block_size]

            # 进行DCT变换
            dct_block = cv2.dct(block)

            # 提取水印 bit
            # 如果系数大于0，判断为1；否则为0
            extracted_bit = 1 if dct_block[4, 4] > 0 else 0
            extracted_watermark[i] = extracted_bit

        # 重塑为原始水印大小
        return extracted_watermark.reshape(self.watermark_size)

    def calculate_similarity(self, original_watermark, extracted_watermark):
        """计算原始水印和提取水印的相似度

        Args:
            original_watermark: 原始水印
            extracted_watermark: 提取的水印

        Returns:
            相似度（0-1之间，值越大越相似）
        """
        # 计算两个水印之间的汉明距离
        hamming_distance = np.sum(np.abs(original_watermark - extracted_watermark))
        # 计算相似度（1 - 错误率）
        similarity = 1 - (hamming_distance / np.prod(self.watermark_size))
        return similarity
