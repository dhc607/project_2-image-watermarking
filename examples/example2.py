import os
import cv2
import matplotlib.pyplot as plt
from src.watermark import Watermark
from src.image_processing import apply_attacks_and_save

# 配置
ORIGINAL_IMAGE = "examples/original.png"
WATERMARKED_IMAGE = "examples/watermarked.png"
ATTACKED_DIR = "examples/attacked"
SECRET_KEY = 12345  # 用于水印嵌入和提取的密钥
WATERMARK_SIZE = (8, 8)  # 水印大小


def create_sample_image():
    """创建示例图像（如果不存在）"""
    if not os.path.exists(ORIGINAL_IMAGE):
        # 创建一个简单的测试图像
        img = cv2.imread("https://picsum.photos/400/400")
        if img is None:  # 如果无法下载图片，创建一个简单的图像
            img = cv2.imread("https://picsum.photos/400/400")
            if img is None:
                img = np.ones((400, 400, 3), dtype=np.uint8) * 255
                cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)

        cv2.imwrite(ORIGINAL_IMAGE, img)
        print(f"已创建示例图像: {ORIGINAL_IMAGE}")


def embed_watermark_demo():
    """演示水印嵌入过程"""
    # 创建水印处理器
    wm = Watermark(secret_key=SECRET_KEY, watermark_size=WATERMARK_SIZE)

    # 生成水印（可以使用自定义消息）
    # 示例：使用自定义消息生成水印
    # message = "MySecretWatermark123"
    # binary_message = ''.join(format(ord(c), '08b') for c in message)
    # watermark = wm.generate_watermark(binary_message[:np.prod(WATERMARK_SIZE)])

    # 生成随机水印
    watermark = wm.generate_watermark()
    print("生成的水印:")
    print(watermark)

    # 嵌入水印
    print("\n正在嵌入水印...")
    watermarked_image = wm.embed(ORIGINAL_IMAGE, watermark, WATERMARKED_IMAGE)

    return wm, watermark


def extract_watermark_demo(wm):
    """演示水印提取过程"""
    print("\n正在从原始带水印图像中提取水印...")
    extracted_watermark = wm.extract(WATERMARKED_IMAGE)
    print("提取的水印:")
    print(extracted_watermark)
    return extracted_watermark


def robustness_test_demo(wm, original_watermark):
    """演示鲁棒性测试"""
    print("\n对带水印图像应用各种攻击...")
    watermarked_image = cv2.imread(WATERMARKED_IMAGE)
    apply_attacks_and_save(watermarked_image, ATTACKED_DIR)

    # 显示几种主要攻击的结果
    print("\n攻击后水印提取结果:")
    key_attacks = ["original", "flip_horizontal", "rotate_30", "jpeg_low", "noise"]

    # 创建一个图形来显示结果
    plt.figure(figsize=(15, 10))

    for i, attack in enumerate(key_attacks):
        attack_image_path = os.path.join(ATTACKED_DIR, f"{attack}.png")
        if os.path.exists(attack_image_path):
            # 读取攻击后的图像
            attacked_image = cv2.imread(attack_image_path)
            attacked_image_rgb = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2RGB)

            # 提取水印
            extracted_watermark = wm.extract(attack_image_path)

            # 计算相似度
            similarity = wm.calculate_similarity(original_watermark, extracted_watermark)

            # 显示图像
            plt.subplot(2, 3, i + 1)
            plt.imshow(attacked_image_rgb)
            plt.title(f"{attack}\n相似度: {similarity:.4f}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("examples/robustness_demo.png")
    print("鲁棒性测试结果已保存至: examples/robustness_demo.png")
    plt.show()


def compare_images():
    """比较原始图像和带水印图像"""
    original = cv2.imread(ORIGINAL_IMAGE)
    watermarked = cv2.imread(WATERMARKED_IMAGE)

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    watermarked_rgb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(watermarked_rgb)
    plt.title("带水印图像")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("examples/comparison.png")
    print("图像对比已保存至: examples/comparison.png")
    plt.show()


def main():
    """主函数"""
    print("=== 数字水印演示程序 ===")

    # 创建示例图像
    create_sample_image()

    # 嵌入水印
    wm, original_watermark = embed_watermark_demo()

    # 提取水印
    extracted_watermark = extract_watermark_demo(wm)

    # 计算相似度
    similarity = wm.calculate_similarity(original_watermark, extracted_watermark)
    print(f"原始水印与提取水印的相似度: {similarity:.4f}")

    # 比较原始图像和带水印图像
    compare_images()

    # 进行鲁棒性测试
    robustness_test_demo(wm, original_watermark)

    print("\n演示完成!")


if __name__ == "__main__":
    main()
