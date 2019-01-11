import unittest
import numpy as np

from multiprocessing import Event, Process, Queue, freeze_support, queues

import cv2

from face_eevee import face_eevee, config, utils

class TestEachProcess(unittest.TestCase):
    def test_dlib(self):
        exit_event = Event()
        in_queue, out_queue = Queue(maxsize=1), Queue(maxsize=1)
        process = face_eevee.Dlib_detector_process(exit_event, in_queue, out_queue, config.p)
        process.start()

        image = cv2.imread('tests/example/image_0010.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        in_queue.put_nowait(gray)
        out = out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        self.assertTrue(process.is_alive())
        exit_event.set()
        process.join()
        self.assertEqual(process.exitcode, 0)

    def test_eos(self):
        exit_event = Event()
        in_queue, out_queue = Queue(maxsize=1), Queue(maxsize=1)
        process = face_eevee.Eos_process(exit_event, in_queue, out_queue, config.share_path, 10, 10)
        process.start()

        shapes = [utils.read_pts('tests/example/image_0010.pts')]

        in_queue.put_nowait(shapes)
        out = out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        self.assertTrue(process.is_alive())
        exit_event.set()
        process.join()
        self.assertEqual(process.exitcode, 0)

    def test_matplotlib(self):
        exit_event = Event()
        in_queue, out_queue = Queue(maxsize=1), Queue(maxsize=1)
        process = face_eevee.Matplotlib_process(exit_event, in_queue, out_queue)
        process.start()

        in_queue.put_nowait(np.zeros((4,3)))
        out = out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        self.assertTrue(process.is_alive())
        exit_event.set()
        process.join()
        self.assertEqual(process.exitcode, 0)


class TestProcessPipeline(unittest.TestCase):
    def test_pipeline(self):
        exit_event = Event()
        dlib_in_queue, dlib_out_queue = Queue(maxsize=1), Queue(maxsize=1)
        dlib_process = face_eevee.Dlib_detector_process(exit_event, dlib_in_queue, dlib_out_queue, config.p)
        dlib_process.start()

        eos_in_queue, eos_out_queue = Queue(maxsize=1), Queue(maxsize=1)
        eos_process = face_eevee.Eos_process(exit_event, eos_in_queue, eos_out_queue, config.share_path, 10, 10)
        eos_process.start()

        matplotlib_in_queue, matplotlib_out_queue = Queue(maxsize=1), Queue(maxsize=1)
        matplotlib_process = face_eevee.Matplotlib_process(exit_event, matplotlib_in_queue, matplotlib_out_queue)
        matplotlib_process.start()

        image = cv2.imread('tests/example/image_0010.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dlib_in_queue.put_nowait(gray)
        out = dlib_out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        eos_in_queue.put_nowait(out)
        out = eos_out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        matplotlib_in_queue.put_nowait(out)
        out = matplotlib_out_queue.get(timeout=3)
        self.assertIsNotNone(out)

        self.assertTrue(dlib_process.is_alive())
        self.assertTrue(eos_process.is_alive())
        self.assertTrue(matplotlib_process.is_alive())
        
        exit_event.set()
        dlib_process.join()
        eos_process.join()
        matplotlib_process.join()

        self.assertEqual(dlib_process.exitcode, 0)
        self.assertEqual(eos_process.exitcode, 0)
        self.assertEqual(matplotlib_process.exitcode, 0)

if __name__ == '__main__':
    unittest.main()