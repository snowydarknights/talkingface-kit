import cv2

def reorder_image(img, input_order='HWC'):
    if input_order == 'CHW':
        img = img.transpose((2, 0, 1))
    elif input_order == 'HWC':
        pass  # 默认已经是HWC格式
    return img

def to_y_channel(img):
    # 将图像从BGR转换为YUV，并提取Y通道
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv_img[:, :, 0]  # 只返回Y通道


def imresize(img, scale=0.5, antialiasing=True):
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)