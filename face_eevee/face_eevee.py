import abc
import copy
import logging
import time
from multiprocessing import Event, Process, Queue, freeze_support, queues

import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import dlib
import eos

from .config import p, share_path




class BaseProcess(Process, abc.ABC):
    def __init__(self, exit_event, in_queue, out_queue):
        super().__init__()
        self.exit = exit_event
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True
        self.name = 'Unimplemented Name'

    def prepare_run(self):
        """Implement this function if there are some initialisation steps before the run method start processing inputs
        
        Returns:
            bool -- Success of the preparation
        """
        return True

    @abc.abstractmethod
    def process_input(self, input_val):
        """Process its inputs and returns the processed value

        Arguments:
            input_val {object} -- input value

        Returns:
            object -- processed output
        """
        return None

    def run(self):
        self.prepare_run()
        while not self.exit.is_set():
            try:
                input_val = self.in_queue.get(block=True, timeout=1)
                output_val = self.process_input(input_val)
                self.out_queue.put_nowait(output_val)
            except queues.Empty:
                pass
            except queues.Full:
                pass
        
        self.in_queue.cancel_join_thread()
        self.out_queue.cancel_join_thread()
        logging.debug('{} exiting'.format(self.name))


class Dlib_detector_process(BaseProcess):
    def __init__(self, exit_event, in_queue, out_queue, p):
        """Face detector process, that use dlib to predicts faces shape

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts input image
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
            p {string} -- path to the dlib model
        """
        super().__init__(exit_event, in_queue, out_queue)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)

    def process_input(self, gray):
        rects = self.detector(gray, 0)
        shapes = []
        for rect in rects:
            # Make the prediction and transfom it to numpy array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            shapes.append(shape)
        return shapes


class Eos_process(BaseProcess):
    def __init__(self, exit_event, in_queue, out_queue, share_path, image_width, image_height):
        """eos lib process, that use eos to fit 3d faces to the landmarks

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts input landmarks
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
            share_path {string} -- path to the folder of eos models
        """
        super().__init__(exit_event, in_queue, out_queue)
        self.share_path = share_path
        self.image_width = image_width
        self.image_height = image_height

    def prepare_run(self):
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

    def process_input(self, shapes):
            for shape in shapes:
                my_landmarks = [eos.core.Landmark(
                    str(idx), xy) for idx, xy in enumerate(shape)]

                (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions,
                                                                                                my_landmarks, self.landmark_mapper, self.image_width, self.image_height,
                                                                                                self.edge_topology, self.contour_landmarks, self.model_contour)

                return np.vstack(mesh.vertices)



class Matplotlib_process(BaseProcess):
    def __init__(self, exit_event, in_queue, out_queue):
        """Matplotlib process, that use matplotlib to plot vertexs

        Arguments:
            in_queue {multiprocessing.Queue} -- Size 1 queue that accepts mesh
            out_queue {multiprocessing.Queue} -- Size 1 queue to put ressults in
        """
        super().__init__(exit_event, in_queue, out_queue)

    def prepare_run(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(15, 135)
    
    def process_input(self, np_mesh):
        self.ax.scatter(np_mesh[:, 0], np_mesh[:, 2], np_mesh[:, 1], s=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.fig.canvas.draw()

        X = np.array(self.fig.canvas.renderer._renderer)
        self.ax.clear()
        return X

def clear_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except queues.Empty:
        pass
    queue.close()

def main():
    first = True

    # initialise placeholders
    rendered_mesh = np.zeros((0, 2, 3))
    shapes = []
    np_mesh = None #np.zeros((1,3))

    # Get first image to set image size
    cap = cv2.VideoCapture(0)
    _, image = cap.read()
    image_width = image.shape[1]
    image_height = image.shape[0]


    # Setup interprocess communication
    # exit_event is shared across all process
    exit_event = Event()

    # Setup de queues for interprocess comm
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

    # setup counters
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

            # Feed the gray scale image to dlib for face detection
            try:
                dlib_in_queue.put_nowait(gray)
            except queues.Full:
                pass

            # Get back the landmarks (skip if not ready, same for all)
            try:
                shapes = dlib_out_queue.get_nowait()
            except queues.Empty:
                pass

            # Feed the landmarks to eos
            try:
                eos_in_queue.put_nowait(shapes)
            except queues.Full:
                pass

            # Get the mesh back
            try:
                np_mesh = eos_out_queue.get_nowait()
            except queues.Empty:
                pass

            # Send the mesh for rendering, only if it exists
            try:
                if np_mesh is not None:
                    matplotlib_in_queue.put_nowait(np_mesh)
            except queues.Full:
                pass

            # Get the render back
            try:
                rendered_mesh = matplotlib_out_queue.get_nowait()
            except queues.Empty:
                pass

            # Draw on our image, all the finded cordinate points (x,y)
            for shape in shapes:
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Make a second image to put the rendered mesh onto
            face_image = np.zeros_like(image)
            face_image[:rendered_mesh.shape[0], :rendered_mesh.shape[1]] = rendered_mesh[:, :, :3]

            # Juxtapose them
            image = np.hstack((image, face_image))

            # Draw fps counter
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,
                        '{:.0f} fps'.format(1/(time.time()-last_time)),
                        (image_width, image_height-50),
                        font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            last_time = time.time()
            
            # Show the image with opencv
            cv2.imshow("Output", image)

            k = cv2.waitKey(5) & 0xFF
            if k in [27, ord('q')]:
                break

        cv2.destroyAllWindows()
    finally:
        # Release all
        cap.release()
        exit_event.set()
        logging.debug('exit_event set')
        
        face_detector_process.join()
        logging.debug('dlib joined')
        eos_process.join()
        logging.debug('eos joined')
        matplotlib_process.join()
        logging.debug('all process joined')
        for queue in [dlib_in_queue, dlib_out_queue, eos_in_queue,
                            eos_out_queue, matplotlib_in_queue, matplotlib_out_queue, ]:
            clear_queue(queue)
        logging.debug('queues cleared')

    


if __name__ == "__main__":
    freeze_support()
    main()

