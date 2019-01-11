import copy
import time
from multiprocessing import Event, Process, Queue, freeze_support, queues, sharedctypes

import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import dlib
import eos


class Dlib_detector_process(Process):
    def __init__(self, exit_event, in_queue, out_queue, p):
        """Face detector process, that use dlib to predicts faces shape

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts input image
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
            p {string} -- path to the dlib model
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True
        self.exit = exit_event

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)

    def run(self):
        while not self.exit.is_set():
            try:
                gray = self.in_queue.get(block=True, timeout=1)

                rects = self.detector(gray, 0)
                shapes = []
                for rect in rects:
                    # Make the prediction and transfom it to numpy array
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    shapes.append(shape)

                self.out_queue.put_nowait(shapes)
            except queues.Empty:
                pass
            except queues.Full:
                pass

        self.in_queue.cancel_join_thread()
        self.out_queue.cancel_join_thread()
        print('Dlib exiting')
        return 0

    def stop(self):
        self.exit.set()


def clear_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except queues.Empty:
        pass
    queue.close()


class Eos_process(Process):
    def __init__(self, exit_event, in_queue, out_queue, share_path, image_width, image_height):
        """eos lib process, that use eos to fit 3d faces to the landmarks

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts input landmarks
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
            share_path {string} -- path to the folder of eos models
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.share_path = share_path
        self.image_width = image_width
        self.image_height = image_height

        self.daemon = True
        self.exit = exit_event

    def run(self):

        self.model = eos.morphablemodel.load_model(
            self.share_path + "/sfm_shape_3448.bin")
        self.blendshapes = eos.morphablemodel.load_blendshapes(
            self.share_path + "/expression_blendshapes_3448.bin")
        # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
        self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(self.model.get_shape_model(), self.blendshapes,
                                                                                 color_model=eos.morphablemodel.PcaModel(),
                                                                                 vertex_definitions=None,
                                                                                 texture_coordinates=self.model.get_texture_coordinates())

        self.landmark_mapper = eos.core.LandmarkMapper(
            self.share_path + '/ibug_to_sfm.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology(
            self.share_path + '/sfm_3448_edge_topology.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(
            self.share_path + '/ibug_to_sfm.txt')
        self.model_contour = eos.fitting.ModelContour.load(
            self.share_path + '/sfm_model_contours.json')

        while not self.exit.is_set():
            try:
                shapes = self.in_queue.get(block=True, timeout=1)
                for shape in shapes:
                    my_landmarks = [eos.core.Landmark(
                        str(idx), xy) for idx, xy in enumerate(shape)]

                    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions,
                                                                                                   my_landmarks, self.landmark_mapper, self.image_width, self.image_height,
                                                                                                   self.edge_topology, self.contour_landmarks, self.model_contour)

                    np_mesh = np.vstack(mesh.vertices)

                    self.out_queue.put_nowait(np_mesh)
            except queues.Empty:
                pass
            except queues.Full:
                pass

        self.in_queue.cancel_join_thread()
        self.out_queue.cancel_join_thread()
        print('Eos exiting')
        return 0

    def stop(self):
        self.exit.set()


class Matplotlib_process(Process):
    def __init__(self, exit_event, in_queue, out_queue):
        """Matplotlib process, that use matplotlib to plot vertexs

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts mesh
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
        """
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True
        self.exit = exit_event

    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(15, 135)
        while not self.exit.is_set():
            try:
                np_mesh = self.in_queue.get(block=True, timeout=1)

                ax.scatter(np_mesh[:, 0], np_mesh[:, 2], np_mesh[:, 1], s=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                fig.canvas.draw()

                X = np.array(fig.canvas.renderer._renderer)
                ax.clear()

                self.out_queue.put_nowait(X)
            except queues.Empty:
                pass
            except queues.Full:
                pass

        self.in_queue.cancel_join_thread()
        self.out_queue.cancel_join_thread()
        print('matplotlib exiting')
        return 0

    def stop(self):
        self.exit.set()


if __name__ == "__main__":
    freeze_support
    share_path = r"C:\4dface\eos\share"
    p = r"C:\4dface\shape_predictor_68_face_landmarks.dat"

    first = True
    render_vertex = True

    X = np.zeros((2, 2, 3))
    shapes = []
    np_mesh = []

    cap = cv2.VideoCapture(0)
    _, image = cap.read()
    image_width = image.shape[1]
    image_height = image.shape[0]

    exit_event = Event()

    dlib_in_queue, dlib_out_queue = Queue(maxsize=1), Queue(maxsize=1)
    face_detector_process = Dlib_detector_process(
        exit_event, dlib_in_queue, dlib_out_queue, p)
    face_detector_process.start()

    eos_in_queue, eos_out_queue = Queue(maxsize=1), Queue(maxsize=1)
    eos_process = Eos_process(
        exit_event, eos_in_queue, eos_out_queue, share_path, image_width, image_height)
    eos_process.start()

    matplotlib_in_queue, matplotlib_out_queue = Queue(
        maxsize=1), Queue(maxsize=1)
    matplotlib_process = Matplotlib_process(
        exit_event, matplotlib_in_queue, matplotlib_out_queue)
    matplotlib_process.start()

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            # Getting out image by webcam
            _, image = cap.read()
            frame_count += 1

            image_width = image.shape[1]
            image_height = image.shape[0]

            # Converting the image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                dlib_in_queue.put_nowait(gray)
            except queues.Full:
                pass

            try:
                shapes = dlib_out_queue.get(block=first, timeout=3)
            except queues.Empty:
                pass

            for shape in shapes:
                # Draw on our image, all the finded cordinate points (x,y)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            try:
                if shapes:
                    eos_in_queue.put_nowait(shapes)
            except queues.Full:
                pass

            try:
                np_mesh = eos_out_queue.get(block=first, timeout=3)
            except queues.Empty:
                pass

            if render_vertex and len(np_mesh) > 0 and not frame_count % 5:
                try:
                    if shapes:
                        matplotlib_in_queue.put_nowait(np_mesh)
                except queues.Full:
                    pass

                try:
                    X = matplotlib_out_queue.get_nowait()
                except queues.Empty:
                    pass
            first = False

            face_image = np.zeros_like(image)
            face_image[:X.shape[0], :X.shape[1]] = X[:, :, :3]

            image = np.hstack((image, face_image))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,
                        '{:.0f} fps'.format(1/(time.time()-last_time)),
                        (image_width, image_height-50),
                        font, 2, (0, 0, 0), 2, cv2.LINE_AA)
            last_time = time.time()
            # Show the image
            cv2.imshow("Output", image)

            k = cv2.waitKey(5) & 0xFF
            if k in [27, ord('q')]:
                break

        cv2.destroyAllWindows()
    finally:
        cap.release()
        exit_event.set()
        print('exit_event set')
        for queue in [dlib_in_queue, dlib_out_queue, eos_in_queue,
                      eos_out_queue, matplotlib_in_queue, matplotlib_out_queue, ]:
            clear_queue(queue)
        face_detector_process.join()
        print('dlib joined')
        eos_process.join()
        print('eos joined')
        matplotlib_process.join()
        print('all process joined')
