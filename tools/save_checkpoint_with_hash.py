import torch
import hashlib
import os

def compute_hash_from_weights(state_dict: dict, hash_length: int = 8) -> str:
    """
    计算 PTH 权重内容的哈希值。
    :param state_dict: PTH 文件中的 state_dict。
    :param hash_length: 哈希值截取长度。
    :return: 计算出的哈希字符串。
    """
    hasher = hashlib.sha256()
    for key, tensor in state_dict.items():
        hasher.update(key.encode())
        hasher.update(tensor.cpu().numpy().tobytes())
    return hasher.hexdigest()[:hash_length]

def resave_pth(input_pth: str, output_pth: str):
    """
    重新加载并保存 PTH 文件，同时在文件名末尾添加基于权重内容的哈希值。
    :param input_pth: 输入的 PTH 文件路径。
    :param output_pth: 重新保存后的 PTH 文件路径。
    """
    # 加载 PTH 文件
    data = torch.load(input_pth, map_location="cpu")
    state_dict = data['state_dict']
    
    # 计算基于权重的哈希值
    file_hash = compute_hash_from_weights(state_dict)
    
    # 解析文件名和扩展名
    base_name, ext = os.path.splitext(output_pth)
    new_output_pth = f"{base_name}-{file_hash}{ext}"  # 添加哈希值
    
    # 重新保存 PTH 文件
    torch.save(state_dict, new_output_pth)
    print(f"PTH 文件已重新保存至: {new_output_pth}")

def check_pth_hash(pth_file: str, expected_hash: str) -> bool:
    """
    检测 PTH 文件的哈希值是否与预期值匹配。
    :param pth_file: PTH 文件路径。
    :param expected_hash: 预期的哈希值。
    :return: 是否匹配。
    """
    # 加载 PTH 文件
    data = torch.load(pth_file, map_location="cpu")
    state_dict = data['state_dict']
    
    # 计算实际哈希值
    actual_hash = compute_hash_from_weights(state_dict)
    
    # 比较哈希值
    if actual_hash == expected_hash:
        print(f"哈希匹配: {actual_hash}")
        return True
    else:
        print(f"哈希不匹配: 预期 {expected_hash}, 实际 {actual_hash}")
        return False


if __name__ == "__main__":
    # 示例使用
    input_pth = "/home/panjiancheng/projects/LAE-DINO/weights/lae_dino_swint_lae_1m_ep26.pth"
    output_pth = "/home/panjiancheng/projects/LAE-DINO/weights/lae_dino_swint_lae1m.pth"
    resave_pth(input_pth, output_pth)