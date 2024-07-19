
class TreeNode(object):
    def __init__(self, ids=None, children=None, entropy=0, depth=0):
        self.ids = ids           # Chỉ mục dữ liệu của nút
        self.entropy = entropy   # Giá trị tính toán entropy
        self.depth = depth       # Khoảng cách đến nút gốc
        self.split_attribute = None  # Thuộc tính được chọn tại nút nếu không phải nút lá
        self.children = children or []  # Danh sách nút con
        self.order = None       
        self.label = None       # Nhãn của nút nếu là nút lá

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute  # Thuộc tính sử dụng
        self.order = order  # Thứ tự của các nút con của nút này

    def set_label(self, label):
        self.label = label  # Gán giá trị nhãn nếu là nút lá