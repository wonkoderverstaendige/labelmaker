import cv2
import math
import time
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

MODES = ['HA', 'HP', 'VA', 'VP']
MODE_COLORS = {'HA': (210, 100, 100), 'HP': (255, 128, 128), 'VA': (100, 210, 100), 'VP': (128, 255, 128)}
COLUMNS = ['presented', 'label',
           'HAx', 'HAy', 'HPx', 'HPy',
           'VAx', 'VAy', 'VPx', 'VPy']


class LabelMaker:
    def __init__(self, vid_path, labels_file=None, start_frame=0, frame_skip=2, force_mode=None):
        self.__start_frame = start_frame

        self.path = Path(vid_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        self.capture = cv2.VideoCapture(str(self.path))

        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame = None
        self.frame_id = None
        self.disp_frame = None

        self.labels_csv_path = Path(labels_file).resolve() if labels_file is not None else None
        if self.labels_csv_path is not None:
            print('Loading existing labels file')
            self.labels = pd.read_csv(str(self.labels_csv_path), index_col='frame')
        else:
            iso_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.labels_csv_path = str(self.path) + '.LabelMaker-{}.csv'.format(iso_date)
            print('Creating empty labels DF')
            self.labels = pd.DataFrame(index=range(self.num_frames),
                                       columns=COLUMNS)
            self.labels['presented'] = 0

        self.alive = True
        self.pressed_key = -1
        self.frame_skip = frame_skip

        self.mouse_x = None
        self.mouse_y = None

        self.mode = 0
        if force_mode is not None:
            print('Forcing mode {}'.format(force_mode))
            self.force_mode = int(force_mode)
            self.mode = self.force_mode
        else:
            self.force_mode = None

        cv2.namedWindow('LabelMaker')
        cv2.setMouseCallback('LabelMaker', self.process_mouse_event)

        self.time_last_grab = None
        self.pbar = tqdm(total=self.num_frames)
        self.loop()

    def loop(self):
        self.grab(absolute=0)
        while self.alive:
            self.disp_frame = self.frame.copy()

            for m in MODES:
                try:
                    m_x = int(self.labels.loc[self.frame_id, m + 'x'])
                    m_y = int(self.labels.loc[self.frame_id, m + 'y'])
                except ValueError:
                    pass
                else:
                    cv2.putText(self.disp_frame, text=m, org=(int(m_x), int(m_y)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1.5,
                                color=MODE_COLORS[m],
                                lineType=cv2.LINE_AA,
                                thickness=2)

            cv2.imshow('LabelMaker', self.disp_frame)
            key = cv2.waitKey(1000 // 30)
            self.process_key(key)

    def grab(self, relative=None, absolute=None):

        # Time to process frame
        # if self.time_last_grab is not None:
            # print('{:.1f} s'.format(time.time() - self.time_last_grab))

        # Frame movements (absolute, relative, next)
        if relative is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.frame_id + relative))
        elif absolute is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, absolute))
        else:
            # skip n frames
            for _ in range(self.frame_skip):
                self.capture.read()
        # read next frame
        rv, frame = self.capture.read()
        self.frame_id = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

        self.pbar.n = self.frame_id
        self.pbar.refresh()

        if rv:
            self.frame = frame
            self.labels.loc[self.frame_id, 'presented'] = 1
            self.time_last_grab = time.time()
        else:
            if self.frame is None:
                raise IOError('Frame acquisition failed')

        self.mode = self.force_mode if self.force_mode else 0

    def process_key(self, key):
        if key < 0:
            return
        self.pressed_key = key

        # Exit with q or Esc
        if key in [27, ord('q')]:
            self.quit()

        # skip frame
        elif key == ord('.'):
            self.grab()

        # skip frame
        elif key == ord(','):
            self.grab(relative=-self.frame_skip - 2)

    def process_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.labels.loc[self.frame_id, '{}x'.format(MODES[self.mode])] = x
            self.labels.loc[self.frame_id, '{}y'.format(MODES[self.mode])] = y
            self.mode = self.mode + 1
            if self.mode >= len(MODES) or self.force_mode is not None:
                self.grab()

    def quit(self):
        self.alive = False
        self.pbar.close()
        cv2.destroyAllWindows()

        # iso_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        # csv_out_path = str(self.path) + '.LabelMaker.csv'.format(iso_date)
        self.labels.to_csv(str(self.labels_csv_path), index_label='frame')
        print('Labels written to {}'.format(self.labels_csv_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-l', '--labels', help='Path to labels csv file to load and update')
    parser.add_argument('-M', '--force_mode', help='Force and process a single mode only')

    cli_args = parser.parse_args()

    LabelMaker(cli_args.path, labels_file=cli_args.labels, force_mode=cli_args.force_mode)


if __name__ == '__main__':
    main()
