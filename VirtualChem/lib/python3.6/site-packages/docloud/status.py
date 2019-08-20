'''The docloud.status module contains classes to store various DOcplexcloud status.
'''

from enum import Enum


class JobExecutionStatus(Enum):
    """ Job execution status values.

    Attributes:
        CREATED: The job has been created.
        NOT_STARTED: The job has been sent to the execution queue and is ready
            to be picked by a worker.
        RUNNING: The job is being executed.
        INTERUPTING: A job interruption has been requested.
        INTERRUPTED: The job has been interrupted.
        FAILED: The job had a system failure.
        PROCESSED: The job has been processed.
    """

    CREATED = 0
    NOT_STARTED = 1
    RUNNING = 2
    INTERRUPTING = 3
    INTERRUPTED = 4
    FAILED = 5
    PROCESSED = 6

    @staticmethod
    def isEnded(status):
        """ Returns true if the specified status means that the job execution ended.

        Job execution is considered ended when it is:

        - FAILED
        - PROCESSED
        - INTERRUPTED

        Args:
            status(JobExecutionStatus): The status to test.

        Returns:
            True if the job execution has ended.
        """
        return (JobExecutionStatus.FAILED == status) or (JobExecutionStatus.PROCESSED == status) \
            or (JobExecutionStatus.INTERRUPTED == status)


class JobSolveStatus(Enum):
    """ Job solve status values.

    This `Enum` is used to convert job solve status string values into an
    enumeration::

        >>> job = client.get_job(jobid)
        >>> solveStatus = JobSolveStatus[job['solveStatus']]

    Attributes:
        UNKNOWN: The algorithm has no information about the solution.
        FEASIBLE_SOLUTION: The algorithm found a feasible solution.
        OPTIMAL_SOLUTION: The algorithm found an optimal solution.
        INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
        UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
        INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
    """
    UNKNOWN = 0
    FEASIBLE_SOLUTION = 1
    OPTIMAL_SOLUTION = 2
    INFEASIBLE_SOLUTION = 3
    UNBOUNDED_SOLUTION = 4
    INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5
