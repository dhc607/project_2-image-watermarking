import unittest
import os
import cv2
import numpy as np
from watermark import Watermark
from image_processing import (
    flip_image, rotate_image, translate_image, crop_image,
    resize_image, adjust_brightness_contrast, add_noise,
    apply_jpeg_compression, apply_blur, apply_attacks_and_save
)

# 测试配置
TEST_IMAGE_PATH = "examples/original.png"
WATERMARKED_IMAGE_PATH = "examples/watermarked.png"
ATTACKED_DIR = "examples/attacked"
SECRET_KEY = 12345
WATERMARK_SIZE = (8, 8)


class TestWatermark(unittest.TestCase):
    """测试水印嵌入和提取功能"""

    @classmethod
    def setUpClass(cls):
        """在所有测试前执行一次，准备测试环境"""
        # 创建示例目录（如果不存在）
        os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
        os.makedirs(ATTACKED_DIR, exist_ok=True)

        # 如果没有测试图像，创建一个简单的测试图像
        if not os.path.exists(TEST_IMAGE_PATH):
            test_img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (50, 50), (200, 200), (0, 255, 0), -1)
            cv2.imwrite(TEST_IMAGE_PATH, test_img)
            print(f"创建测试图像: {TEST_IMAGE_PATH}")

        # 初始化水印处理器
        cls.wm = Watermark(secret_key=SECRET_KEY, watermark_size=WATERMARK_SIZE)

        # 生成测试水印
        cls.original_watermark = cls.wm.generate_watermark()

        # 嵌入水印
        cls.watermarked_image = cls.wm.embed(
            TEST_IMAGE_PATH,
            cls.original_watermark,
            WATERMARKED_IMAGE_PATH
        )

        # 对带水印的图像应用各种攻击
        apply_attacks_and_save(cls.watermarked_image, ATTACKED_DIR)

    def test_embedding_extraction(self):
        """测试水印嵌入和提取功能"""
        # 从带水印的图像中提取水印
        extracted_watermark = self.wm.extract(WATERMARKED_IMAGE_PATH)

        # 计算相似度
        similarity = self.wm.calculate_similarity(self.original_watermark, extracted_watermark)
        print(f"\n原始图像水印提取相似度: {similarity:.4f}")

        # 验证相似度（未受攻击的情况下应该非常高）
        self.assertGreater(similarity, 0.95)

    def test_robustness(self):
        """测试水印的鲁棒性"""
        print("\n鲁棒性测试结果:")
        results = []

        # 测试所有攻击后的图像
        for attack_name in os.listdir(ATTACKED_DIR):
            if attack_name.endswith(".png"):
                attack_type = os.path.splitext(attack_name)[0]
                image_path = os.path.join(ATTACKED_DIR, attack_name)

                # 提取水印
                extracted_watermark = self.wm.extract(image_path)

                # 计算相似度
                similarity = self.wm.calculate_similarity(
                    self.original_watermark,
                    extracted_watermark
                )

                results.append((attack_type, similarity))
                print(f"{attack_type}: 相似度 = {similarity:.4f}")

        # 对关键攻击类型进行验证
        # 这些阈值是经验值，实际应用中可能需要调整
        for attack_type, similarity in results:
            if attack_type == "original":
                self.assertGreater(similarity, 0.95)
            elif attack_type in ["flip_horizontal", "flip_vertical", "rotate_30"]:
                self.assertGreater(similarity, 0.7)
            elif attack_type in ["brightness_up", "brightness_down", "contrast_high", "contrast_low"]:
                self.assertGreater(similarity, 0.8)
            elif attack_type in ["jpeg_low", "noise"]:
                self.assertGreater(similarity, 0.6)


if __name__ == "__main__":
    unittest.main()
