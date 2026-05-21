import sys
import struct
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QSplitter, QScrollArea, QComboBox, QGroupBox,
    QCheckBox, QAction, QMessageBox, QMenu
)
from PyQt5.QtCore import Qt, QFile, QIODevice, QDataStream, QPoint, QRegExp
from PyQt5.QtGui import QRegExpValidator, QRegExpValidator, QIcon

INIT0 = [
    ("Flag", [(4, ("Type", ["Grid GM Init", "ASID Init", "Task"])),
              (2, ("Dependency", ["No Dependency",
                                  "Group Dependency", "All Dependency"])),
              (1, ("Group End", None)),
              (1, ("Grid End", None)),
              (1, ("Pool End", None)),
              (13, ("RSD", None)),
              (1, ("Flush L2D", None)),
              ]),
    ("Next", "#x"),
    ("Group numbers", None),
    ("RSD", None),
    ("Interrupt", [
        (1, ("Grid Done", None)),
        (2, ("RSD", None)),
        (1, ("GM Fault", None)),
    ]),
    ("ID", [(16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("GM remap", None),
    ("GM sync", None),
    ("GM address low", "#x"),
    ("GM address high", "#x"),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("RSD", None),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
]
INIT1 = [
    ("Flag", [(4, ("Type", ["Grid GM Init", "ASID Init", "Task"])),
              (2, ("Dependency", ["No Dependency",
                                  "Group Dependency", "All Dependency"])),
              (1, ("Group End", None)),
              (1, ("Grid End", None)),
              (1, ("Pool End", None)),
              (12, ("RSD", None)),
              (1, ("Grid Init Flag", None)),
              ]),
    ("Core MMU Seg Ctrl", None),
    ("Core Remap Ctrl0", None),
    ("Core Remap Ctrl1", None),
    ("Interrupt", [
        (1, ("Group(Task) Done", None))
    ]),
    ("ID", [(16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("Core Seg 0 Ctrl 0", "#x"),
    ("Core Seg 0 Ctrl 1", "#x"),
    ("Core Seg 1 Ctrl 0", "#x"),
    ("Core Seg 1 Ctrl 1", "#x"),
    ("Core Seg 2 Ctrl 0", "#x"),
    ("Core Seg 2 Ctrl 1", "#x"),
    ("Core Seg 3 Ctrl 0", "#x"),
    ("Core Seg 3 Ctrl 1", "#x"),
    ("Core Seg 4 Ctrl 0", "#x"),
    ("Core Seg 4 Ctrl 1", "#x"),
    ("Core Seg 5 Ctrl 0", "#x"),
    ("Core Seg 5 Ctrl 1", "#x"),
    ("Core Seg 6 Ctrl 0", "#x"),
    ("Core Seg 6 Ctrl 1", "#x"),
    ("Core Seg 7 Ctrl 0", "#x"),
    ("Core Seg 7 Ctrl 1", "#x"),
    ("ASID 0 Low / Control", "#x"),
    ("ASID 0 High", "#x"),
    ("ASID 1 Low / Control", "#x"),
    ("ASID 1 High", "#x"),
    ("ASID 2 Low / Control", "#x"),
    ("ASID 2 High", "#x"),
    ("ASID 3 Low / Control", "#x"),
    ("ASID 3 High", "#x"),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
]
TASK = [
    ("Flag", [(4, ("Type", ["Grid GM Init", "ASID Init", "Task"])),
              (2, ("Dependency", ["No Dependency",
                                  "Group Dependency", "All Dependency"])),
              (1, ("Group End", None)),
              (1, ("Grid End", None)),
              (1, ("Pool End", None)),
              ]),
    ("Next", "#x"),
    ("RSD Loop counter", None),  # loop counter
    ("Start PC", "#x"),
    ("Interrupt", [
        (1, ("TEC Done", None)),
        (1, ("TEC Signal IRQ", None)),
        (1, ("TEC Exception", None)),
        (1, ("TEC Fault", None)),
    ]),
    ("ID", [(16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("ID/CTRL", [(16, ("Task ID", None)), (8, ("Warm Length", None))]),
    ("Dim", [(16, ("Grid Dim X", None)), (16, ("Grid Dim Y", None))]),
    ("Dim", [(16, ("Grid Dim Z", None)), (16, ("Group Dim X", None))]),
    ("Dim", [(16, ("Group Dim Y", None)), (16, ("Group Dim Z", None))]),
    ("Id", [(16, ("Group Id X", None)), (16, ("Group Id Y", None))]),
    ("Id", [(16, ("Group Id Z", None)), (16, ("Task Id X", None))]),
    ("Dim", [(16, ("Task Id Y", None)), (16, ("Task Id Z", None))]),
    ("Stake Pointer", "#x"),
    ("Parameter Pointer", "#x"),
    ("Data Pointer", "#x"),
    ("Constant Pointer", "#x"),
    ("Printer Buffer Pointer", "#x"),
    ("Profiler Buffer Pointer", "#x"),
    ("Id", [(16, ("Core ID", None)), (16, ("Cluster ID", None))]),
    ("Tec ID", None),
    ("RSD FMDP", None),
    ("RSD TAP", None),
    ("RSD DAP", None),
    ("RSD PAP", None),
    ("RSD IDP", None),
    ("Data Size", "#x"),
    ("TCB Pointer", "#x"),
    ("RSD", None),
    ("RSD", None),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
    ("Group Dependency", [
        (16, ("Group ID", None)), (16, ("Grid ID", None))]),
]


def _mask_update(old, new, mask):
    # update value with mask
    return (old & ~mask) | (new & mask)


class BinaryParserApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_title = "TCB文件解析器"
        self.setWindowTitle(self.window_title)
        self.current_file_path = ""
        self.setGeometry(100, 100, 900, 600)
        self.setAcceptDrops(True)

        self.fields = [INIT0, INIT1, TASK]

        self.blocks = []  # 存储解析后的块数据
        self.initUI()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 检查是否为文件
            event.acceptProposedAction()  # 接受拖拽[5](@ref)
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()  # 获取本地路径
            self.blocks += self.parse_binary_file(file_path)
            self.update_block_list()

            if self.blocks:
                self.status_bar.showMessage(
                    f"成功解析: {file_path} ({len(self.blocks)}个块)", 3000)
                self.block_list.setCurrentRow(0)
            else:
                self.status_bar.showMessage("无法解析文件或文件格式错误", 3000)

    def on_rows_moved(self, sourceParent, start, end, destinationParent, destinationRow):
        # list move item
        moves = self.blocks[start:end+1]
        self.blocks = self.blocks[:start]+self.blocks[end+1:]
        self.blocks = self.blocks[:destinationRow]+moves+self.blocks[destinationRow:]

    def _value_changed(self, i, offset, new_value, mask=0xffffffff):
        old_value = self.blocks[i][offset]
        if len(new_value) != 0:
            new_value = new_value.lower().strip()
            if new_value.startswith("0x"):
                if len(new_value) > 2:
                    new_value = int(new_value, 16)  # 将十六进制字符串转换为整数
                    self.blocks[i][offset] = _mask_update(
                        self.blocks[i][offset], new_value, mask)
            elif new_value.startswith("0b"):
                if len(new_value) > 2:
                    new_value = int(new_value, 2)  # 将二进制字符串转换为整数
                    self.blocks[i][offset] = _mask_update(
                        self.blocks[i][offset], new_value, mask)
            else:
                new_value = int(new_value)  # 将输入的值转换为整数
                self.blocks[i][offset] = _mask_update(
                    self.blocks[i][offset], new_value, mask)
        print(
            f"blocks[{i}][{offset}] updated from {old_value:#x} to {self.blocks[i][offset]:#x}")

    def _bit_changed(self, i, offset, new_value, mask=0xffffffff):
        old_value = self.blocks[i][offset]

        def get_shift(mas):
            i = 0
            while mas & 1 == 0:
                i += 1
                mas >>= 1
            return i
        if isinstance(new_value, bool):
            new_value = int(new_value)
        if isinstance(new_value, str) and len(new_value) != 0:
            new_value = new_value.lower().strip()
            if new_value.startswith("0x"):
                if len(new_value) > 2:
                    new_value = int(new_value, 16)  # 将十六进制字符串转换为整数
                else:
                    new_value = None
            elif new_value.startswith("0b"):
                if len(new_value) > 2:
                    new_value = int(new_value, 2)  # 将二进制字符串转换为整数
                else:
                    new_value = None
            else:
                new_value = int(new_value)  # 将输入的值转换为整数
        if isinstance(new_value, int):
            new_value = new_value << get_shift(mask)
            self.blocks[i][offset] = _mask_update(
                self.blocks[i][offset], new_value, mask)
        print(
            f"blocks[{i}][{offset}] updated from {old_value:#x} to {self.blocks[i][offset]:#x}")
        if offset == 0:  # type changed
            self.display_block_details(i)

    def create_field_widget(self, field_desc, value, i, offset, msk=0xffffffff):
        name = field_desc[0]
        desc = field_desc[1]
        if name.startswith("RSD"):  # 跳过RSD字段的创建
            return None
        field_widget = QWidget()
        hbox = QHBoxLayout(field_widget)

        def make_changed(i, offset, msk, func=self._bit_changed):
            return lambda x: func(i, offset, x, msk)
        if desc is None or isinstance(desc, str):
            # 基本字段：标签+文本框
            label = QLabel(f"{name}:")
            is_align = isinstance(desc, str) and ":align" in desc
            if isinstance(desc, str) and desc.startswith("#x"):
                value = f"{value:#x}"
            else:
                value = str(value)
            line_edit = QLineEdit(value)
            line_edit.setMaxLength(12)
            regex = QRegExp(r'^(0x[0-9A-Fa-f]+|0b[01]+|\d+)$')
            validator = QRegExpValidator(regex, line_edit)
            line_edit.setValidator(validator)
            if is_align:
                line_edit.textEdited.connect(make_changed(
                    i, offset, msk, self._value_changed))
            else:
                line_edit.textEdited.connect(make_changed(i, offset, msk))
            hbox.addWidget(label)
            hbox.addWidget(line_edit)

        elif isinstance(desc, list):
            if isinstance(desc[0], tuple):
                # 分段字段：分组容器
                group = QGroupBox(name)
                layout = QGridLayout()
                # 位偏移计算
                bit_offset = 0
                num_wid = 0
                for bit_width, sub_desc in desc:
                    # 提取子字段值
                    mask = (1 << bit_width) - 1
                    sub_value = (value >> bit_offset) & mask
                    msk = mask << bit_offset
                    bit_offset += bit_width
                    if bit_width == 1:
                        # 位字段：复选框
                        wid = QCheckBox(sub_desc[0])
                        wid.setChecked(sub_value == 1)

                        wid.clicked.connect(make_changed(i, offset, msk))
                    else:
                        # 递归创建子控件
                        wid = self.create_field_widget(
                            sub_desc, sub_value, i, offset, msk)
                    if wid is not None:
                        layout.addWidget(wid, num_wid // 4, num_wid % 4)
                        num_wid += 1

                group.setLayout(layout)
                hbox.addWidget(group)
            else:
                # 枚举字段：下拉框
                label = QLabel(f"{name}:")
                combo = QComboBox()
                combo.addItems(desc)
                combo.setCurrentIndex(value)
                combo.currentIndexChanged.connect(make_changed(i, offset, msk))
                # combo.setEnabled(False)
                hbox.addWidget(label)
                hbox.addWidget(combo)

        return field_widget

    def create_menu_bar(self):
        """创建菜单栏及其动作"""
        menu_bar = self.menuBar()

        # 文件菜单
        file_menu = menu_bar.addMenu("文件(&F)")

        # 打开动作
        open_action = QAction(QIcon.fromTheme(
            "document-open"), "打开(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        open_action.setStatusTip("打开二进制文件")
        file_menu.addAction(open_action)

        # 保存动作
        save_action = QAction(QIcon.fromTheme(
            "document-save"), "保存(&S)...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        save_action.setStatusTip("保存解析结果")
        file_menu.addAction(save_action)

        # 添加分隔线
        file_menu.addSeparator()

        # 退出动作
        exit_action = QAction(QIcon.fromTheme(
            "application-exit"), "退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        exit_action.setStatusTip("退出应用程序")
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("编辑(&E)")
        add_action = QAction(QIcon.fromTheme(
            "document-open"), "添加文件(&A)...", self)
        add_action.setShortcut("Ctrl+A")
        add_action.triggered.connect(self.add_file)
        add_action.setStatusTip("添加TCB文件")
        edit_menu.addAction(add_action)

        update_action = QAction(QIcon.fromTheme(
            "document-edit"), "Update Next Address(&U)...", self)
        update_action.setShortcut("Ctrl+U")
        update_action.triggered.connect(self.update_next_address)
        update_action.setStatusTip("Update Next Address")
        edit_menu.addAction(update_action)

        ids_action = QAction(QIcon.fromTheme(
            "document-edit"), "Update IDs(&I)...", self)
        ids_action.setShortcut("Ctrl+I")
        ids_action.triggered.connect(self.update_ids)
        ids_action.setStatusTip("Update Group/Grid/Task ID")
        edit_menu.addAction(ids_action)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助(&H)")

        # 关于动作
        about_action = QAction(QIcon.fromTheme(
            "help-about"), "关于(&A)...", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_action.setStatusTip("显示程序信息")
        help_menu.addAction(about_action)

    def open_file(self):
        """打开文件对话框并加载文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开TCB文件", "", "TCB文件 (*.tcb*);;所有文件 (*)"
        )

        if file_path:
            self.status_bar.showMessage(f"正在解析: {file_path}", 5000)
            QApplication.processEvents()  # 更新UI

            # 读取并解析文件
            self.blocks = self.parse_binary_file(file_path)
            self.update_block_list()

            if self.blocks:
                self.status_bar.showMessage(
                    f"成功解析: {file_path} ({len(self.blocks)}个块)", 3000)
                self.block_list.setCurrentRow(0)
                self.current_file_path = file_path
                self.setWindowTitle(f"{self.window_title} - {file_path}")
            else:
                self.status_bar.showMessage("无法解析文件或文件格式错误", 3000)

    def add_file(self):
        """打开文件对话框并加载文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开TCB文件", "", "TCB文件 (*.tcb*);;所有文件 (*)"
        )

        if file_path:
            self.status_bar.showMessage(f"正在解析: {file_path}", 5000)
            QApplication.processEvents()  # 更新UI

            # 读取并解析文件
            self.blocks += self.parse_binary_file(file_path)
            self.update_block_list()

            if self.blocks:
                self.status_bar.showMessage(
                    f"成功解析: {file_path} ({len(self.blocks)}个块)", 3000)
                self.block_list.setCurrentRow(0)
            else:
                self.status_bar.showMessage("无法解析文件或文件格式错误", 3000)

    def save_file(self, path=None):
        """保存文件功能"""
        if not self.current_file_path:
            QMessageBox.warning(self, "保存失败", "请先打开一个文件")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存解析结果", "", "TCB文件 (*.tcb*);;所有文件 (*)"
        )

        if file_path:
            # 这里添加实际保存逻辑
            try:
                # 示例保存逻辑 - 实际应保存解析结果
                with open(file_path, 'wb') as f:
                    for i, block in enumerate(self.blocks):
                        f.write(struct.pack(f'<{len(block)}I', *tuple(block)))

                self.status_bar.showMessage(f"文件已保存到: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存文件时出错:\n{str(e)}")

    def show_about_dialog(self):
        """显示关于对话框"""
        about_text = (
            "TCB Editor\n\n"
            "Version: 1.0\n"
            "Author: Neo\n"
            "Parsing and editing TCB files\n\n"
            "© 2025 版权所有"
        )
        QMessageBox.about(self, "关于", about_text)

    def update_next_address(self):
        """更新下一个地址"""
        base_offset = -1
        for i, block in enumerate(self.blocks):
            t = self.get_block_type(block)
            if t != 1 and i != len(self.blocks) - 1:
                if base_offset == -1:
                    base_offset = block[1]
                else:
                    value = block[1]
                    correct = base_offset + i * 0x80
                    if value != correct:
                        self.status_bar.showMessage(
                            f"Corret Next Addrs of block {i} from {value:#x} to {correct:#x}", 1000)
                        self.blocks[i][1] = correct
                        if i == self.block_list.currentRow():
                            self.display_block_details(i)
                if t == 2:  # task tcb, update tcbp
                    value = block[27]
                    correct = base_offset + (i - 1) * 0x80
                    if value != correct:
                        self.status_bar.showMessage(
                            f"Corret Next Addrs of block {i} from {value:#x} to {correct:#x}", 1000)
                        self.blocks[i][27] = correct
                        if i == self.block_list.currentRow():
                            self.display_block_details(i)

    def update_ids(self):
        """更新Grid/Group ID"""
        grid = 0
        group = 0
        task = 0
        for i, block in enumerate(self.blocks):
            end = self.get_end_type(block)
            t = self.get_block_type(block)
            if grid == -1:
                ids = block[5]
                group = ids & 0xffff
                grid = ids >> 16
                continue
            ids = (grid & 0xffff) << 16 | (group & 0xffff)
            self.blocks[i][5] = ids
            if t == 2:
                ids2 = block[6]
                self.blocks[i][6] = (ids2 & 0xffff0000) | (task & 0xffff)
                task += 1
            if t == 0:
                group += 1
            if end & 0b1:  # group end
                group += 1
                task = 0
            if end & 0b10:  # grid end
                grid += 1

    def initUI(self):
        self.create_menu_bar()
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 顶部按钮布局
        # top_layout = QHBoxLayout()

        # 主内容布局（使用分割器）
        content_layout = QHBoxLayout()

        # Block列表（左侧）
        self.block_list = QListWidget()
        self.block_list.currentRowChanged.connect(self.display_block_details)
        self.block_list.setMinimumWidth(100)
        self.block_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.block_list.setDefaultDropAction(Qt.DropAction.MoveAction)  # 确保移动而非复制
        self.block_list.setAcceptDrops(True)
        self.block_list.model().rowsMoved.connect(self.on_rows_moved)

        # 字段展示区域（右侧）
        self.field_container = QWidget()
        self.field_layout = QVBoxLayout(self.field_container)
        self.field_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 添加滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.field_container)

        # 使用分割器布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.block_list)
        splitter.addWidget(scroll_area)
        splitter.setSizes([150, 700])

        content_layout.addWidget(splitter)
        # main_layout.addLayout(top_layout)
        main_layout.addLayout(content_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪", 3000)

    def parse_binary_file(self, file_path):
        """解析二进制文件"""
        blocks = []

        file = QFile(file_path)
        if not file.open(QIODevice.OpenModeFlag.ReadOnly):
            return blocks

        # 计算文件大小和块数量
        file_size = file.size()
        if file_size % 128 != 0:
            print(f"警告: 文件大小({file_size}字节)不是128字节的整数倍")

        block_count = file_size // 128

        # 使用QDataStream读取数据
        data_stream = QDataStream(file)
        data_stream.setByteOrder(
            QDataStream.ByteOrder.LittleEndian)  # 根据实际情况调整字节序

        for _ in range(block_count):
            block = []
            for i in range(32):
                # 读取32位无符号整数
                value = data_stream.readUInt32()
                block.append(value)

            blocks.append(block)

        file.close()
        return blocks

    def update_block_list(self):
        """更新块列表"""
        self.block_list.clear()
        tcb_type = ["Init TCB", "ASID TCB", "Task TCB", ]+["Unknown TCB"]*16
        for i, block in enumerate(self.blocks):
            # 显示块的关键信息
            flag_value = block[0]  # Flag字段
            next_value = block[1]   # Next字段

            t = tcb_type[self.get_block_type(block)]
            self.block_list.addItem(
                f"{i:<3} {t:>10} (Flag: 0x{flag_value:X}, Next: 0x{next_value:X})")

    def get_block_type(self, block):
        return block[0] & 0b1111

    def get_end_type(self, block):
        return (block[0] >> 6) & 0b111

    def display_block_details(self, block_index):
        """显示选定块的详细信息（改用QLabel+QLineEdit）"""
        # 清空现有字段
        while self.field_layout.count():
            child = self.field_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if block_index < 0 or block_index >= len(self.blocks):
            return

        block = self.blocks[block_index]
        try:
            fields = self.fields[self.get_block_type(block)]
        except IndexError:
            self.blocks = []
            self.status_bar.showMessage("Invalid block type")
            QMessageBox.critical(
                self, "Error", "Invalid block type, Please make sure your data is TCB format.")
            return

        # 为每个字段创建显示控件
        for i, value in enumerate(block):
            # 创建水平布局容器
            if i < len(fields):
                field_widget = self.create_field_widget(
                    fields[i], value, block_index, i)
                if field_widget is not None:
                    self.field_layout.addWidget(field_widget)


def main():
    app = QApplication(sys.argv)
    window = BinaryParserApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
