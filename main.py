import sys
from os.path import expanduser
import faulthandler

from PyQt5 import QtCore, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import math
import keyboard


OS = sys.platform


def read_qss(filepath: str):
    with open(filepath, 'r') as style:
        return style.read()


def is_video_file(path: str):
    try:
        name = path.split('/')[-1]
        extension = name.split('.')[-1]
        if extension in ['mp4', 'mkv']:
            return 1
        return 0
    except Exception as error:
        print(error)
        return -1


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setFont(QFont('Times', 10))
        # ---------------------------------------
        self.stack = QStackedWidget(self)
        self.stack.addWidget(RecordWindow(self))
        self.stack.addWidget(TemplateWindow(self))
        self.stack.addWidget(BoundingBoxWindow(self))
        # ---------------------------------------
        self.title = 'Template Creator'
        self.geometry = (1280, 480, 670, 380)
        # ---------------------------------------
        self.init_header()
        self.setCentralWidget(self.stack)
        # ---------------------------------------
        self.show()

    def init_header(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.geometry[0], self.geometry[1], self.geometry[2], self.geometry[3])
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')
        edit_menu = main_menu.addMenu('Edit')
        view_menu = main_menu.addMenu('Windows')
        search_menu = main_menu.addMenu('Search')
        tools_menu = main_menu.addMenu('Tools')
        help_menu = main_menu.addMenu('Help')

        record_btn = QAction('Record Video', self)
        record_btn.setShortcut('Ctrl+R')
        record_btn.triggered.connect(lambda: self.set_window(0))
        view_menu.addAction(record_btn)

        template_btn = QAction('Template', self)
        template_btn.setShortcut('Ctrl+T')
        template_btn.triggered.connect(lambda: self.set_window(1))
        view_menu.addAction(template_btn)

        bounding_box_btn = QAction('Bounding Box', self)
        bounding_box_btn.setShortcut('Ctrl+B')
        bounding_box_btn.triggered.connect(lambda: self.set_window(2))
        view_menu.addAction(bounding_box_btn)

        home_btn = QAction('Home', self)
        home_btn.setShortcut('Ctrl+H')
        home_btn.triggered.connect(lambda: self.set_window(0))
        view_menu.addAction(home_btn)

        exit_btn = QAction(QIcon('exit24.png'), 'Exit', self)
        exit_btn.setShortcut('Ctrl+Q')
        exit_btn.setStatusTip('Exit application')
        exit_btn.triggered.connect(self.close)
        file_menu.addAction(exit_btn)

        self.show()

    def set_window(self, index):
        self.stack.setCurrentIndex(index)


class TemplateWindow(QWidget):
    def __init__(self, parent: MainWindow):
        super(TemplateWindow, self).__init__(parent)
        self.container = QHBoxLayout(self)
        self.container.setContentsMargins(0, 0, 0, 0)
        self.container.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)

        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet(read_qss('./style/template.qss'))
        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop)

        self.main_frame = QFrame(self)
        self.main_frame.setStyleSheet('QFrame{background-color:rgb(195, 130, 127);}')
        self.main_menu = QVBoxLayout(self.main_frame)
        self.main_menu.setContentsMargins(0, 0, 0, 0)
        self.main_menu.setSpacing(0)

        # Widgets

        # row 0

        self.category_text = QLabel('Category:')
        self.category = QLineEdit("")

        # row 1

        self.input_text = QLabel('Input File:')
        self.file = QPushButton('undefined')
        self.file.clicked.connect(self.select_file)
        self.file.setShortcut('Ctrl+I')

        # - row 2

        self.output_text = QLabel('Output Folder:')
        self.out_folder = QPushButton(expanduser('~').split('/')[-1])
        self.out_folder.setToolTip(expanduser("~"))
        self.out_folder.clicked.connect(self.select_folder)
        self.out_folder.setShortcut('Ctrl+O')

        # row 3

        self.write_btn = QPushButton('Write All')
        self.write_btn.clicked.connect(self.write)

        self.write_frame_btn = QPushButton('Write Frame')
        self.next_frame_btn = QPushButton('Next')
        self.prev_frame_btn = QPushButton('Prev')

        # attach side menu
        self.side_menu.addWidget(self.category_text, 0, 0)
        self.side_menu.addWidget(self.category, 0, 1)
        self.side_menu.addWidget(self.input_text, 1, 0)
        self.side_menu.addWidget(self.file, 1, 1)
        self.side_menu.addWidget(self.output_text, 2, 0)
        self.side_menu.addWidget(self.out_folder, 2, 1)
        self.side_menu.addWidget(self.write_btn, 3, 0, 1, 2)
        self.side_menu.addWidget(self.write_frame_btn, 4, 0, 1, 0)
        self.side_menu.addWidget(self.prev_frame_btn, 5, 0)
        self.side_menu.addWidget(self.next_frame_btn, 5, 1)

        # main menu
        self.video = None
        self.preview = QLabel()
        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.main_menu.addWidget(self.preview)
        self.main_menu.addWidget(self.progress)

        # final
        self.splitter.addWidget(self.side_frame)
        self.splitter.addWidget(self.main_frame)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([175, 175])

        self.container.addWidget(self.splitter)

    def select_file(self):
        file_dir = QFileDialog.getOpenFileName(self, 'Select File', expanduser('~'), "Image files (*.mp4)")
        self.file.setText(file_dir[0].split('/')[-1])
        self.file.setToolTip(file_dir[0])
        self.video = cv2.VideoCapture(file_dir[0])
        try:
            _, frame = self.video.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.preview.setPixmap(QPixmap.fromImage(
                QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).scaled(640, 480,
                                                                                                Qt.KeepAspectRatio)))
        except Exception as error:
            print(error)

    def select_folder(self):
        out_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.out_folder.setToolTip(out_dir)
        self.out_folder.setText(str(out_dir.split('/')[-1]))

    def write(self):
        self.extractor = VideoExtractor(self)
        self.extractor.start()


class BoundingBoxWindow(QWidget):
    start_point_signal = pyqtSignal(list)
    end_point_signal = pyqtSignal(list)
    mouse_point_signal = pyqtSignal(list)
    update_signal = pyqtSignal(bool)

    def __init__(self, parent):

        super(BoundingBoxWindow, self).__init__(parent)
        self.splitter = QSplitter(Qt.Horizontal)
        self.container = QHBoxLayout(self)
        self.container.setContentsMargins(0, 0, 0, 0)
        self.container.setSpacing(0)

        # side frame
        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet(read_qss('./style/default.qss'))
        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop)

        # main frame
        self.main_frame = QFrame(self)

        # main menu
        self.preview = QLabel(self.main_frame)
        self.preview.setAlignment(Qt.AlignTop)
        self.preview.setStyleSheet('QFrame{background-color:rgb(195, 130, 127);}')

        self.preview.setMouseTracking(True)
        self.preview.mouseMoveEvent = self.set_mouse_position
        self.preview.mousePressEvent = self.set_start_position
        # self.preview.mouseReleaseEvent = self.set_end_position

        # side menu
        self.file_text = QLabel('Video File:')
        self.file = QPushButton('undefined')
        self.file.clicked.connect(self.select_file)
        self.file.setShortcut('Ctrl+F')

        self.output_text = QLabel('Output File:')
        self.output_dir = QPushButton(expanduser('~').split('/')[-1])
        self.output_dir.clicked.connect(self.select_dir)
        self.output_dir.setShortcut('Ctrl+O')

        self.write_btn = QPushButton('write')
        self.prev_btn = QPushButton('previous')
        self.next_btn = QPushButton('next')
        self.next_btn.clicked.connect(lambda: self.feed.next_frame())
        self.next_btn.setShortcut('n')
        # progress
        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.info_divider = QLabel('INFO:')
        self.info_divider.setStyleSheet('border-right-width:0;border-bottom-width:2px;border-style:solid;')

        # INFO
        self.point_1 = QLabel('Point A: (0, 0)')
        self.point_1.setStyleSheet('background-color:rgba(255, 0, 0, 0.25);border-width:0;')
        self.point_2 = QLabel('Point B: (0, 0)')
        self.point_2.setStyleSheet('background-color:rgba(255, 0, 0, 0.25);border-width:0;')

        self.side_menu.addWidget(self.file_text, 0, 0)
        self.side_menu.addWidget(self.file, 0, 1)
        self.side_menu.addWidget(self.output_text, 1, 0)
        self.side_menu.addWidget(self.output_dir, 1, 1)
        self.side_menu.addWidget(self.write_btn, 2, 0)
        # self.side_menu.addWidget(self.prev_btn, 3, 0)
        self.side_menu.addWidget(self.next_btn, 2, 1)
        self.side_menu.addWidget(self.progress, 3, 0, 1, 0)

        # INFO - Widgets
        self.side_menu.addWidget(self.info_divider, 4, 0, 1, 2)
        # self.side_menu.addWidget(self.point_1, 5, 0)
        # self.side_menu.addWidget(self.point_2, 6, 0)
        # self.side_menu.addWidget()

        self.splitter.addWidget(self.side_frame)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([150, 150])

        self.container.addWidget(self.splitter)
        self.show()
        # feed
        self.feed = BoundingBox(self)
        self.feed.progress_update.connect(lambda e: self.progress.setValue(int(e)))

        # other
        if OS not in ['linux', 'linux2']:
            self.hook = keyboard.on_press(self.keyboard_listener)

    def __del__(self):
        self.feed.stop()

    def keyboard_listener(self, event):
        if event.event_type == 'down':
            if event.name == 'n':
                self.feed.next_frame()
            elif event.name == 'left':
                pass

    def set_mouse_position(self, event):
        if event.buttons() & Qt.LeftButton:
            end_point = [min(max(event.x(), 0), self.feed.shape[1]), min(max(event.y(), 0), self.feed.shape[0])]
            self.end_point_signal.emit(end_point)
        mouse_pos = [event.x(), event.y()]
        self.mouse_point_signal.emit(mouse_pos)
        self.update_signal.emit(True)

    def set_start_position(self, event):
        start_pos = [max(event.x(), 0), max(event.y(), 0)]
        self.start_point_signal.emit(start_pos)
        # self.point_1.setText(f'Point A: ({event.x()}, {event.y()})')
        # self.feed.update = True
        self.update_signal.emit(True)

    def set_end_position(self, event):
        self.end_point_signal.emit([min(max(event.x(), 0), self.feed.shape[1]), min(max(event.y(), 0), self.feed.shape[0])])
        # self.point_2.setText(f'Point B: ({event.x()}, {event.y()})')
        # self.feed.update = True
        self.update_signal.emit(True)

    def start(self):
        self.feed.start()
        self.feed.ImageUpdate.connect(self.update_image)
        self.feed.next_frame()

    def update_image(self, image):
        self.preview.setPixmap(QPixmap.fromImage(image))

    def select_file(self):
        file_dir = QFileDialog.getOpenFileName(self, 'Select File', expanduser('~'), "Image files (*.mp4)")
        if file_dir[0] == '': return
        self.file.setText(file_dir[0].split('/')[-1])
        self.file.setToolTip(file_dir[0])
        self.feed.set_video(file_dir[0])
        self.start()

    def select_dir(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.output_dir.setText(input_dir.split('/')[-1])
        self.output_dir.setToolTip(input_dir)
        self.feed.path = self.output_dir.toolTip()


class BoundingBox(QtCore.QThread):
    ImageUpdate = pyqtSignal(QImage)
    progress_update = pyqtSignal(object)

    def __init__(self, parent: BoundingBoxWindow):
        super(BoundingBox, self).__init__(parent)
        self.parent: BoundingBoxWindow = parent
        self.is_active = False
        self.update = False
        self.video: cv2.VideoCapture = None
        self.image = None
        self.path: str = expanduser("~")
        print(self.path)
        # bounding box attributes
        self.p1: list = [0, 0]
        self.p2: list = [0, 0]
        self.mouse: list = [0, 0]
        self.shape: list = [0, 0]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.index = 0
        self.video_count = 0
        # py signals
        self.parent.start_point_signal.connect(lambda e: self._set_p1(list(e)))
        self.parent.end_point_signal.connect(lambda e: self._set_p2(list(e)))
        self.parent.mouse_point_signal.connect(lambda e: self._set_mouse(list(e)))
        self.parent.update_signal.connect(lambda e: self._set_update(bool(e)))

    def _set_p1(self, val: list): self.p1 = val
    def _set_p2(self, val: list): self.p2 = val
    def _set_mouse(self, val: list): self.mouse = val
    def _set_update(self, val: bool): self.update = val

    def set_video(self, file: str):
        if is_video_file(file):
            self.video = cv2.VideoCapture(file)
            self.shape = [int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          3]
            self.video_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def next_frame(self):
        if self.video is not None:
            ret, image = self.video.read()
            self.index += 1
            if ret:
                self.progress_update.emit(int((100 / self.video_count) * self.index))
                cv2.imwrite(f'{self.path}/[{self.p1[0], self.p1[1]}, {self.p2[0], self.p2[1]}].png', image)
                self.image = image
                self.update = True
            else:
                self.video.release()
                self.video = None
                self.image = np.full((self.shape[0], self.shape[1], self.shape[2]), 255, dtype=np.uint8)
                self.image = cv2.copyMakeBorder(self.image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None,
                                                value=[150, 150, 150])
                self.image = cv2.putText(img=self.image, text='No Video!', org=(20, 80), fontFace=self.font,
                                         fontScale=2, color=[0, 0, 0])
                qt_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                                  QImage.Format_RGB888)
                self.ImageUpdate.emit(qt_image.scaled(640, 480, Qt.KeepAspectRatio))
                self.progress_update.emit(0)
                self.stop()
                return

    def run(self):
        self.is_active = True
        image = None
        while self.is_active:
            if not self.update or self.image is None:
                continue

            frame = cv2.copyTo(self.image, None, image)
            frame = cv2.cvtColor(src=frame, dst=frame, code=cv2.COLOR_BGR2RGB)

            frame = cv2.line(img=frame, pt1=[self.mouse[0], 0], pt2=[self.mouse[0], self.shape[0]],
                             color=[255, 255, 255])

            frame = cv2.line(img=frame, pt1=[0, self.mouse[1]], pt2=[self.shape[1], self.mouse[1]],
                             color=[255, 255, 255])

            image = cv2.rectangle(frame, self.p1, self.p2, color=[255, 0, 0], thickness=2)
            for point in [self.p1, self.p2]:
                image = cv2.putText(img=image, text=f'({point[0]},{point[1]})', org=(point[0], point[1] + 20),
                                    fontFace=self.font, fontScale=0.5, thickness=1, color=[255, 255, 255])
                image = cv2.circle(image, point, radius=3, color=[0, 255, 0], thickness=3)

            qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(qt_image)
            self.update = False

    def stop(self):
        self.is_active = False


class RecordWindow(QWidget):
    def __init__(self, parent):
        super(RecordWindow, self).__init__(parent)
        self.container = QHBoxLayout(self)
        self.container.setContentsMargins(0, 0, 0, 0)
        self.container.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.is_recording = False

        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet(read_qss('./style/default.qss'))

        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop)

        self.main_frame = QFrame(self)
        self.main_frame.setStyleSheet('QFrame{background-color:rgb(195, 130, 127);}')
        self.main_menu = QVBoxLayout(self.main_frame)
        self.main_menu.setContentsMargins(0, 0, 0, 0)
        self.main_menu.setSpacing(0)

        # self.info_menu = QGridLayout(self.side_frame)
        # self.info_menu.setAlignment(Qt.AlignBottom)
        # self.info_menu.addWidget(QPushButton('Test'))

        # row 0
        self.file_txt = QLabel('File Name:')
        self.filename = QLineEdit('output.mp4')
        self.filename.textChanged[str].connect(self.handle_footage)

        # row 1
        self.folder_txt = QLabel('Output Folder:')
        self.folder = QPushButton(expanduser('~').split('/')[-1])
        self.folder.setToolTip(expanduser('~'))

        self.folder.clicked.connect(self.select_dir)
        self.folder.setShortcut('Ctrl+O')

        # row  2
        self.start_btn = QPushButton('Start Feed')
        self.start_btn.clicked.connect(self.start)

        self.record_btn = QPushButton('Start Recording')
        self.record_btn.clicked.connect(self.toggle_record)
        self.record_btn.setDisabled(True)

        # row 3
        self.snapshot_btn = QPushButton('Take Snapshot')
        self.snapshot_btn.setDisabled(True)
        self.snapshot_btn.clicked.connect(self.take_snapshot)

        # main window
        self.feed_label = QLabel(self.main_frame)
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed = LiveFeed(self)

        self.main_menu.addWidget(self.feed_label)

        # add widgets
        self.side_menu.addWidget(self.file_txt, 0, 0)
        self.side_menu.addWidget(self.filename, 0, 1)
        self.side_menu.addWidget(self.folder_txt, 1, 0)
        self.side_menu.addWidget(self.folder, 1, 1)
        self.side_menu.addWidget(self.start_btn, 2, 0)
        self.side_menu.addWidget(self.record_btn, 2, 1)
        self.side_menu.addWidget(self.snapshot_btn, 3, 0, 1, 2)

        self.test = QSlider(Qt.Horizontal)
        self.test.setMinimum(0)
        self.test.setMaximum(60)
        self.side_menu.addWidget(self.test, 4, 0, 1, 2)


        # final
        self.splitter.addWidget(self.side_frame)
        self.splitter.addWidget(self.main_frame)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([175, 175])

        self.container.addWidget(self.splitter)

    def __del__(self):
        self.cancel()

    def handle_footage(self, event):
        if not self.feed.cap.isOpened():
            return
        filetype: str = event.split('.')[-1]
        if filetype in ['mp4']:
            self.record_btn.setDisabled(False)
        else:
            self.record_btn.setDisabled(True)

    def take_snapshot(self):
        print(f'{self.folder.toolTip()}/{self.filename.text().split(".")[0]}.png')
        cv2.imwrite(f'{self.folder.toolTip()}/{self.filename.text()}', img=self.feed.frame)

    def get_mouse_pos(self, event):
        position: tuple = (event.pos().x(), event.pos().y())
        self.feed.position = position

        if cv2.waitKey(1) & 0xFF == ord('w'):
            self.feed.box_size = (self.feed.box_size[0] + 10, self.feed.box_size[1])
        elif cv2.waitKey(1) & 0xFF == ord('h'):
            self.feed.box_size = (self.feed.box_size[0], self.feed.box_size[1] + 10)

    def start(self):
        self.record_btn.setDisabled(False)
        self.snapshot_btn.setDisabled(False)
        self.start_btn.setText('Stop')
        self.start_btn.clicked.connect(self.cancel)
        self.feed.start()
        self.feed.ImageUpdate.connect(self.update_image)

    def select_dir(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.folder.setText(str(input_dir).split('/')[-1])
        self.folder.setToolTip(str(input_dir))
        del self.feed
        self.feed = LiveFeed(self)
        self.feed.path = input_dir

    def toggle_record(self):
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.record_btn.setText('Stop Recording')
        else:
            self.record_btn.setText('Start Recording')

    def update_image(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def cancel(self):
        self.start_btn.setText('Start')
        self.start_btn.clicked.connect(self.start)
        self.record_btn.setDisabled(True)
        self.snapshot_btn.setDisabled(True)
        self.feed.stop()


class VideoExtractor(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent: TemplateWindow):
        super().__init__(parent)
        self.parent: TemplateWindow = parent
        self.cap: cv2.VideoCapture = parent.video
        self.category: str = parent.category.text()
        self.path: str = parent.out_folder.toolTip()
        self.is_active: bool = False
        self.idx: int = 0
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def run(self):
        self.is_active = True
        print(f'{self.path}/{self.category}_{self.idx}.png')
        while self.cap.isOpened() and self.is_active:
            ret, frame = self.cap.read()
            if ret:
                image = cv2.flip(frame, 1)
                cv2.imwrite(f'{self.path}/{self.category}_{self.idx}.png', image)
            else:
                self.is_active = False
            self.idx += 1
            self.parent.progress.setValue(int())
            self.parent.progress.setValue(int((100 / self.frame_count) * self.idx))
        self.cap.release()
        print('[INFO]: Finished!')

    def stop(self):
        self.is_active = False


class LiveFeed(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)
        try:
            self.cap: cv2.VideoCapture = cv2.VideoCapture()
            self.path = parent.folder.toolTip()
            self.filename = parent.filename.text()
            self.is_active = False
            self.font = cv2.FONT_HERSHEY_SIMPLEX

            # bounding box
            self.position: tuple = (20, 20)
            self.box_size: tuple = (60, 60)

            self.frame: np.array = None

        except Exception as error:
            print(error)

    def __del__(self):
        self.cap.release()
        print("CALLED DESTRUCTOR")

    def run(self):
        self.is_active = True
        self.cap = cv2.VideoCapture(0)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{self.path}/{self.filename}', fourcc, 30.0, (int(w), int(h)))
        if self.cap is None:return
        while self.is_active:
            ret, self.frame = self.cap.read()
            if ret:
                image = cv2.flip(self.frame, 1)

                if self.parent().is_recording:
                    out.write(image)
                    image = cv2.putText(image, "RECORDING", color=[0, 0, 255], fontScale=1, thickness=2,
                                        fontFace=self.font, org=(0, 25))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, value=[150, 150, 150])

                qt_image = QImage(image.data, image.shape[1], image.shape[0],
                                  QImage.Format_RGB888)
                Pic = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
        empty = np.full((int(h), int(w), 3), 255, dtype=np.uint8)
        empty = cv2.putText(empty, "NO VIDEO!", color=[0, 0, 0], fontScale=1, fontFace=self.font, thickness=3,
                            org=(0, 25))
        empty = cv2.copyMakeBorder(empty, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, value=[150, 150, 150])
        self.ImageUpdate.emit(QImage(empty.data, empty.shape[1], empty.shape[0],
                                     QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio))
        self.cap.release()
        out.release()

    def stop(self):
        self.is_active = False


def main():
    faulthandler.enable()
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    _app = MainWindow()
    _app.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
