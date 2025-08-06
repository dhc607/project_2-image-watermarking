import cv2
import numpy as np
import os


def flip_image(image, flip_code=1):
    """翻转图像

    Args:
        image: 输入图像（ndarray）
        flip_code: 翻转方式，1表示水平翻转，0表示垂直翻转，-1表示水平垂直同时翻转

    Returns:
        翻转后的图像
    """
    return cv2.flip(image, flip_code)


def rotate_image(image, angle=30):
    """旋转图像

    Args:
        image: 输入图像（ndarray）
        angle: 旋转角度（度）

    Returns:
        旋转后的图像
    """
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def translate_image(image, tx=50, ty=50):
    """平移图像

    Args:
        image: 输入图像（ndarray）
        tx: x方向平移像素数
        ty: y方向平移像素数

    Returns:
        平移后的图像
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))


def crop_image(image, x=50, y=50, width=200, height=200):
    """裁剪图像

    Args:
        image: 输入图像（ndarray）
        x: 起始x坐标
        y: 起始y坐标
        width: 裁剪宽度
        height: 裁剪高度

    Returns:
        裁剪后的图像
    """
    rows, cols = image.shape[:2]
    # 确保裁剪区域在图像范围内
    x = max(0, min(x, cols - 1))
    y = max(0, min(y, rows - 1))
    width = min(width, cols - x)
    height = min(height, rows - y)

    return image[y:y + height, x:x + width]


def resize_image(image, scale=0.5):
    """缩放图像

    Args:
        image: 输入图像（ndarray）
        scale: 缩放比例

    Returns:
        缩放后的图像
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """调整图像亮度和对比度

    Args:
        image: 输入图像（ndarray）
        brightness: 亮度调整值（-100到100）
        contrast: 对比度调整值（0.0到3.0）

    Returns:
        调整后的图像
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def add_noise(image, noise_level=0.01):
    """为图像添加高斯噪声

    Args:
        image: 输入图像（ndarray）
        noise_level: 噪声水平（0.0到1.0）

    Returns:
        添加噪声后的图像
    """
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * noise_level * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image, quality=50):
    """对图像应用JPEG压缩

    Args:
        image: 输入图像（ndarray）
        quality: JPEG质量（0到100，值越低压缩率越高）

    Returns:
        压缩后的图像
    """
    # 使用cv2的imencode模拟JPEG压缩
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def apply_blur(image, ksize=5):
    """对图像应用模糊处理

    Args:
        image: 输入图像（ndarray）
        ksize: 模糊核大小

    Returns:
        模糊后的图像
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def save_image(image, output_path):
    """保存图像

    Args:
        image: 要保存的图像（ndarray）
        output_path: 保存路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"已保存图像至: {output_path}")


# 攻击函数映射表，用于批量测试
ATTACK_FUNCTIONS = {
    "original": lambda x: x,  # 原始图像，无攻击
    "flip_horizontal": lambda x: flip_image(x, 1),  # 水平翻转
    "flip_vertical": lambda x: flip_image(x, 0),  # 垂直翻转
    "rotate_30": lambda x: rotate_image(x, 30),  # 旋转30度
    "rotate_90": lambda x: rotate_image(x, 90),  # 旋转90度
    "translate": lambda x: translate_image(x, 30, 30),  # 平移
    "crop": lambda x: crop_image(x, 20, 20, x.shape[1] - 40, x.shape[0] - 40),  # 裁剪
    "resize_half": lambda x: resize_image(x, 0.5),  # 缩小到一半
    "resize_double": lambda x: resize_image(x, 2.0),  # 放大一倍
    "brightness_up": lambda x: adjust_brightness_contrast(x, 50, 1.0),  # 增加亮度
    "brightness_down": lambda x: adjust_brightness_contrast(x, -50, 1.0),  # 降低亮度
    "contrast_high": lambda x: adjust_brightness_contrast(x, 0, 1.5),  # 增加对比度
    "contrast_low": lambda x: adjust_brightness_contrast(x, 0, 0.5),  # 降低对比度
    "noise": lambda x: add_noise(x, 0.02),  # 添加噪声
    "jpeg_low": lambda x: apply_jpeg_compression(x, 30),  # 低质量JPEG压缩
    "blur": lambda x: apply_blur(x, 5)  # 模糊处理
}


def apply_attacks_and_save(original_image, output_dir):
    """对图像应用所有攻击并保存结果

    Args:
        original_image: 原始图像（ndarray）
        output_dir: 输出目录
    """
    for attack_name, attack_func in ATTACK_FUNCTIONS.items():
        try:
            attacked_image = attack_func(original_image)
            output_path = os.path.join(output_dir, f"{attack_name}.png")
            save_image(attacked_image, output_path)
        except Exception as e:
            print(f"应用{attack_name}攻击时出错: {str(e)}")
