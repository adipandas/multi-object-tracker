import numpy as np


class KalmanFilter:
    """
    Kalman Filter Implementation.

    Parameters
    ----------
    transition_matrix: numpy.ndarray
        Transition matrix of shape (n, n).
    measurement_matrix: numpy.ndarray
        Measurement matrix of shape (m, n).
    control_matrix: numpy.ndarray
        Control matrix of shape (m, n).
    process_noise_covariance: numpy.ndarray
        Covariance matrix of shape (n, n).
    measurement_noise_covariance: numpy.ndarray
        Covariance matrix of shape (m, m).
    prediction_covariance: numpy.ndarray
        Predicted (a priori) estimate covariance of shape (n, n).
    initial_state: numpy.ndarray
        Initial state of shape (n,).

    """
    def __init__(
        self,
        transition_matrix,
        measurement_matrix,
        control_matrix=None,
        process_noise_covariance=None,
        measurement_noise_covariance=None,
        prediction_covariance=None,
        initial_state=None
    ):

        self.state_size = transition_matrix.shape[1]
        self.observation_size = measurement_matrix.shape[1]

        self.transition_matrix = transition_matrix
        self.measurement_matrix = measurement_matrix

        self.control_matrix = 0 if control_matrix is None else control_matrix

        self.process_covariance = np.eye(self.state_size) \
            if process_noise_covariance is None else process_noise_covariance

        self.measurement_covariance = np.eye(self.observation_size) \
            if measurement_noise_covariance is None else measurement_noise_covariance

        self.prediction_covariance = np.eye(self.state_size) if prediction_covariance is None else prediction_covariance

        self.x = np.zeros((self.state_size, 1)) if initial_state is None else initial_state

    def predict(self, u=0):
        self.x = np.dot(self.transition_matrix, self.x) + np.dot(self.control_matrix, u)

        self.prediction_covariance = np.dot(
            np.dot(self.transition_matrix, self.prediction_covariance), self.transition_matrix.T
        ) + self.process_covariance

        return self.x

    def update(self, z):
        y = z - np.dot(self.measurement_matrix, self.x)

        innovation_covariance = np.dot(
            self.measurement_matrix, np.dot(self.prediction_covariance, self.measurement_matrix.T)
        ) + self.measurement_covariance

        optimal_kalman_gain = np.dot(
            np.dot(self.prediction_covariance, self.measurement_matrix.T),
            np.linalg.inv(innovation_covariance)
        )

        self.x = self.x + np.dot(optimal_kalman_gain, y)
        eye = np.eye(self.state_size)
        _t1 = eye - np.dot(optimal_kalman_gain, self.measurement_matrix)
        t1 = np.dot(np.dot(_t1, self.prediction_covariance), _t1.T)
        t2 = np.dot(np.dot(optimal_kalman_gain, self.measurement_covariance), optimal_kalman_gain)
        self.prediction_covariance = t1 + t2


class KFTracker:
    """
    1-dimensional Kalman Filter considering Constant Acceleration motion model.

    Parameters
    ----------
    time_step : int
        Number of steps per iteration or number of frames skipped for each step.
    """

    def __init__(self, time_step=1):
        self.time_step = time_step
        self.kf = None

    def setup(
            self,
            process_noise_covariance=None,
            measurement_noise_covariance=None,
            initial_state=None
    ):
        """
        Setup and initialize Kalman Filter.

        Parameters
        ----------
        process_noise_covariance : float or numpy.ndarray
            Process noise covariance matrix of shape (3, 3) or covariance magnitude as scalar value.
        measurement_noise_covariance : float or numpy.ndarray
            Measurement noise covariance matrix of shape (1,) or covariance magnitude as scalar value.
        initial_state : numpy.ndarray
            Initial state of the tracker.

        Returns
        -------

        """
        if process_noise_covariance is None:
            process_noise_covariance = np.array([
                [0.25 * self.time_step ** 4, 0.5 * self.time_step ** 3, 0.5 * self.time_step ** 2],
                [0.5 * self.time_step ** 3, self.time_step ** 2, self.time_step],
                [0.5 * self.time_step ** 2, self.time_step, 1]
            ])
        else:
            if not np.isscalar(process_noise_covariance):
                assert len(process_noise_covariance.shape) == 2
                assert process_noise_covariance.shape[0] == 3 and process_noise_covariance.shape[1] == 3
            else:
                process_noise_covariance = process_noise_covariance * np.array([
                    [0.25 * self.time_step ** 4, 0.5 * self.time_step ** 3, 0.5 * self.time_step ** 2],
                    [0.5 * self.time_step ** 3, self.time_step ** 2, self.time_step],
                    [0.5 * self.time_step ** 2, self.time_step, 1]
                ])

        if measurement_noise_covariance is None:
            measurement_noise_covariance = np.array([1.])
        else:
            if not np.isscalar(measurement_noise_covariance):
                assert len(measurement_noise_covariance.shape) == 1
                assert measurement_noise_covariance.shape[0] == 1
            else:
                measurement_noise_covariance = measurement_noise_covariance * np.array([1.])

        if initial_state is None:
            initial_state = 0.1 * np.random.randn(3)
        elif np.isscalar(initial_state):
            initial_state = np.array([initial_state, 0., 0.])       # assume only initial velocity is known.
        else:
            assert len(initial_state.shape) == 1, len(initial_state.shape)
            assert initial_state.shape[0] == 3, initial_state.shape[0]

        transition_matrix = np.array([[1., self.time_step, self.time_step * self.time_step * 0.5],
                                      [0., 1., self.time_step],
                                      [0., 0., 1.]])

        measurement_matrix = np.array([[1., 0, 0]])

        prediction_noise_covariance = 1. * np.ones((3, 3))

        self.kf = KalmanFilter(
            transition_matrix=transition_matrix,
            measurement_matrix=measurement_matrix,
            process_noise_covariance=process_noise_covariance,
            measurement_noise_covariance=measurement_noise_covariance,
            prediction_covariance=prediction_noise_covariance,
            initial_state=initial_state
        )
        
    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)


class KFTracker2D:
    def __init__(self, time_step=1):
        self.kf_x = KFTracker(time_step=time_step)
        self.kf_y = KFTracker(time_step=time_step)

    def setup(
            self,
            process_noise_covariance_x=None,
            measurement_noise_covariance_x=None,
            initial_state_x=None,
            process_noise_covariance_y=None,
            measurement_noise_covariance_y=None,
            initial_state_y=None
    ):
        if process_noise_covariance_x is not None and process_noise_covariance_y is None:
            process_noise_covariance_y = process_noise_covariance_x

        if measurement_noise_covariance_x is not None and measurement_noise_covariance_y is None:
            measurement_noise_covariance_y = measurement_noise_covariance_x

        if initial_state_x is not None and initial_state_y is None:
            initial_state_y = initial_state_x

        self.kf_x.setup(
            process_noise_covariance=process_noise_covariance_x,
            measurement_noise_covariance=measurement_noise_covariance_x,
            initial_state=initial_state_x
        )

        self.kf_y.setup(
            process_noise_covariance=process_noise_covariance_y,
            measurement_noise_covariance=measurement_noise_covariance_y,
            initial_state=initial_state_y
        )

    def predict(self):
        x_prediction = self.kf_x.predict()
        y_prediction = self.kf_y.predict()
        return np.array([x_prediction[0], y_prediction[0]])

    def update(self, measurement):
        self.kf_x.update(measurement[0])
        self.kf_y.update(measurement[1])


def sort_transition_matrix(t):
    a = np.array([[1., t, t * t * 0.5], [0., 1., t], [0., 0., 1.]])
    z = np.zeros((3, 3))
    a1, a2, a3 = np.concatenate([a, z, z], axis=1), np.concatenate([z, a, z], axis=1), np.concatenate([z, z, a], axis=1)
    a = np.concatenate([a1, a2, a3, np.zeros((1, 9))], axis=0)
    transition_matrix = np.concatenate([a, np.zeros((10, 1))], axis=1)
    assert transition_matrix.shape == (10, 10)
    return transition_matrix


def sort_measurement_matrix():
    m = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.]
    ])
    return m


def sort_process_noise_covariance_matrix(t):
    a = np.array([[0.25 * t ** 4, 0.5 * t ** 3, 0.5 * t ** 2], [0.5 * t ** 3, t ** 2, t], [0.5 * t ** 2, t, 1]])
    z = np.zeros((3, 3))
    a1, a2, a3 = np.concatenate([a, z, z], axis=1), np.concatenate([z, a, z], axis=1), np.concatenate([z, z, a], axis=1)
    a = np.concatenate([a1, a2, a3, np.zeros((1, 9))], axis=0)
    cov = np.concatenate([a, np.zeros((10, 1))], axis=1)
    return cov


class KFTrackerSORT:
    def __init__(self, time_step=1):
        self.time_step = time_step

    def setup(self, process_noise_covariance=None, measurement_noise_covariance=None, initial_state=None):
        """
        Setup and initialize Kalman Filter.

        Parameters
        ----------
        process_noise_covariance : float or numpy.ndarray
            Process noise covariance matrix of shape (3, 3) or covariance magnitude as scalar value.
        measurement_noise_covariance : float or numpy.ndarray
            Measurement noise covariance matrix of shape (1,) or covariance magnitude as scalar value.
        initial_state : numpy.ndarray
            Initial state of the tracker as tuple
            `(x_c, x_c_d, x_c_dd, y_c, y_c_d, y_c_dd, w_c, w_c_d, w_c_dd, aspect_ratio)`.

        Returns
        -------
        """

        if process_noise_covariance is None:
            process_noise_covariance = sort_process_noise_covariance_matrix(self.time_step)
        else:
            if not np.isscalar(process_noise_covariance):
                assert len(process_noise_covariance.shape) == 2
                assert process_noise_covariance.shape[0] == 10 and process_noise_covariance.shape[1] == 10
            else:
                process_noise_covariance = (
                        process_noise_covariance * sort_process_noise_covariance_matrix(self.time_step)
                )

        if measurement_noise_covariance is None:
            measurement_noise_covariance = np.eye(4)
        else:
            if not np.isscalar(measurement_noise_covariance):
                assert len(measurement_noise_covariance.shape) == 2
                assert measurement_noise_covariance.shape[0] == 4 and measurement_noise_covariance.shape[1] == 4
            else:
                measurement_noise_covariance = measurement_noise_covariance * np.eye(4)

        prediction_noise_covariance = np.ones((10, 10))

        if initial_state is None:
            initial_state = np.random.randn(10)
        else:
            assert len(initial_state.shape) == 1
            assert initial_state.shape[0] == 10

        self.kf = KalmanFilter(
            transition_matrix=sort_transition_matrix(self.time_step),
            measurement_matrix=sort_measurement_matrix(),
            process_noise_covariance=process_noise_covariance,
            measurement_noise_covariance=measurement_noise_covariance,
            prediction_covariance=prediction_noise_covariance,
            initial_state=initial_state
        )

    def update(self, measurement):
        """
        Measurement for the filter.

        Parameters
        ----------
        measurement : numpy.ndarray
            Bounding box tuple as `(x_centroid, y_centroid, width, aspect_ratio)`.

        Returns
        -------

        """
        self.kf.update(measurement)

    def predict(self):
        """
        Predict bounding box location.

        Returns
        -------
        bbox : numpy.ndarray
            bounding box tuple as `(x_centroid, y_centroid, width, aspect_ratio)`
        """
        x_state = self.kf.predict()
        x = np.array((x_state[0], x_state[3], x_state[6], x_state[9]))
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def create_data(t=1000, prediction_noise=1, measurement_noise=1,
                    non_linear_input=True, velocity_scale=1/200.):

        x = np.zeros((t,))

        if non_linear_input:
            vel = np.array([np.sin(i * np.pi * velocity_scale) for i in range(t)])
        else:
            vel = np.array([0.001*i for i in range(t)])
        vel_noise = vel + np.random.randn(t) * prediction_noise

        x_noise = np.zeros((t,))
        x_measure_noise = np.random.randn(t) * measurement_noise

        x_noise[0] = 0.
        x_measure_noise[0] += x_noise[0]

        for i in range(t):
            x[i] = x[i-1] + vel[i-1]

            x_noise[i] = x[i-1] + vel_noise[i-1]

            x_measure_noise[i] += x_noise[i]

        return x, vel, x_noise, vel_noise, x_measure_noise

    t = 1000
    x, vel, x_noise, vel_noise, x_measure_noise = create_data(t=t)

    kf = KFTracker(time_step=1)
    kf.setup(measurement_noise_covariance=100, process_noise_covariance=10)

    x_prediction = [np.array([x_measure_noise[0], vel_noise[0], 0.]).flatten()]

    for i in range(1, t):
        x_prediction.append(kf.predict())
        kf.update(x_measure_noise[i])

    x_prediction = np.array(x_prediction)

    time = np.arange(t)
    a = [time, x, '-', time, x_measure_noise, '--', time, x_prediction[:, 0], '-.']
    # a = [time, x, '-', time, x_prediction[:, 0], '-.']
    plt.plot(*a)
    plt.legend(['true', 'noise', 'kf'])
    plt.xlim([0, t])
    plt.grid(True)
    plt.show()
