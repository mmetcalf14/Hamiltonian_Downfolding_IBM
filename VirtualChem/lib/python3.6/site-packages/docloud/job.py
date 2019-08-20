from docloud.status import JobExecutionStatus

import json
import gettext
import gzip as gz
import os
import sys
import tempfile
import time
import datetime
import shutil
from io import BytesIO

import requests

import six
from six import iteritems, string_types

try:
    import pandas
except ImportError:
    pandas = None


# initialize gettext
localedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locale")
translation = gettext.translation('docloud', localedir=localedir, fallback=True)
if sys.version_info[0] == '2':
    _ = translation.ugettext
else:
    _ = translation.gettext


def default_solution_name(parameters):
    # Returns the name of the default solution output attachment for
    # solver engines. Might be different for workers that publish multiple
    # outputs
    solution_attachment_ext = "json"  # default
    if parameters is not None:
        if 'oaas.resultsFormat' in parameters:
            solution_attachment_ext = parameters['oaas.resultsFormat'].lower()
            if solution_attachment_ext == 'text':
                solution_attachment_ext = 'txt'
    return "solution.%s" % solution_attachment_ext


def download_and_format_log_items(client, jobid, seqid, log_file):
    logs = client.get_log_items(jobid, seqid, True)
    for log in logs:
        seqid = log['seqid'] + 1
        for r in log['records']:
            message = r['message'].rstrip()
            log_file.write('%s\n' % message)
    return seqid


class JobAttachmentInfo(object):
    """JobAttachmentInfo is used to create an attachment.

    When creating a JobAttachmentInfo, you provide a ``dict`` which can contain
    the following keys:

    - name: contains the name of the attachment.
    - filename: contains the local filename for the attachment.
    - data: contains raw data.
    - file: contains a file where to read the data.

    Only one of 'filename', 'data' or 'file' can be specified.

    Example: ``{ 'name' : 'diet.lp' , 'filename' : 'data/models/diet.lp' }``


    Args:
        att_info: a ``dict`` containing job attachment info.
        file: if ``att_info`` contains only the 'name' attribute, the file
            object containing the data to be read.
        data: if ``att_info`` contains only the 'name' attribute, the data for
            this attachment.
        filename: if ``att_info`` contains only the 'name' attribute, the name
            of the file to be read.

    Raises:
        ParameterError -- if more than one of file, filename or data are specified

    """

    def __init__(self, att_info, file=None, data=None, filename=None):
        """ Creates a new attachment info.

        Only one of filename, data or file can be used at a time.

        Args:
            att_info: a ``dict`` containing job attachment info.
            file: if ``att_info`` contains only the 'name' attribute, the file
                object containing the data to be read.
            data: if ``att_info`` contains only the 'name' attribute, the data
                for this attachment.
            filename: if ``att_info`` contains only the 'name' attribute, the
                name of the file to be read.

        Raises:
            ParameterError -- if more than one of file, filename or data are
                specified.

        """
        self.file = file
        self.filename = filename
        self.data = data
        self._my_file = None
        if (type(att_info) is dict):
            self.__build_from_dict(att_info)
        elif isinstance(att_info, six.string_types):
            # in the case this is a file that exists
            if os.path.isfile(att_info):
                basename = os.path.basename(att_info)
                self.name = basename
                self.filename = att_info
            else:
                raise ParameterError(_("JobClient.uploadAttachment(): File not found \"%s\"") % att_info)
        # if data is a file-like (has a read()) method, just set it to file
        if self.data and hasattr(self.data, 'read'):
            self.file = self.data
            self.data = None
        # Check that this is well constructed and that only one of file, data
        # or filename is set
        notNone = int(self.file is not None) + int(self.data is not None) \
            + int(self.filename is not None)
        if (notNone != 1):
            raise ParameterError(_("JobClient.uploadAttachment() needs one of (data, file, filename) parameters"))

    def __build_from_dict(self, obj):
        self.name = obj['name']
        file = obj.get('file', self.file)
        data = obj.get('data', self.data)
        if pandas and isinstance(data, pandas.DataFrame):
            # as we always convert to csv, no need to check extension on name
            file = tempfile.TemporaryFile('w+b')
            z = data.to_csv(index=False)  # no way to really force pandas.to_csv() to write bytes...
            file.write(z.encode('utf-8'))
            file.flush()
            file.seek(0)
            data = None  # we replace data with file containing the csv
        self.file = file
        self.filename = obj.get('filename', self.filename)
        self.data = data

    def get_data(self):
        """ Returns the data for this attachment.

        If attachment data was specified as raw data, return the raw data.
        Otherwise if attachment file was specified, call ``read()`` on the file and
        return the data.
        Finally, if attachment filename was specified, open the file, read and
        return the data.

        Returns:
            The data as a byte array.
        """
        data = self.data
        if self.filename is not None:
            with open(self.filename, "rb") as f:
                data = f.read()
        if self.file is not None:
            data = self.file.read()
        return data

    def _get_data_or_file(self):
        """Returns the data for this attachment, or the file like object if
        it is a file like object.

        Returns:
            is_data, data_or_file -- ``is_data`` is a boolean indicating if 
                ``data_or_file`` is a data block.
        """
        # user provided data
        if self.data is not None:
            return True, self.data
        # user provided file
        if self.file is not None:
            return False, self.file
        # if we already opened a file, return it
        if self._my_file is not None:
            return False, self._my_file
        # user provided filename -> open file
        if self.filename is not None:
            self._my_file = open(self.filename, "rb")
            return False, self._my_file

    def get_data_or_file(self, gzip=False):
        """Returns the data for this attachment, or the file like object if
        it is a file like object.

        Args:
            gzip: if True, then the data is gzipped before returned.
        """
        is_data, data_or_file = self._get_data_or_file()  
        if gzip:
            gzipped = tempfile.TemporaryFile("w+b")
            with gz.GzipFile(fileobj=gzipped, mode="wb") as f:
                if is_data:
                    f.write(data_or_file)
                else:
                    shutil.copyfileobj(data_or_file, f)
            # reset position so that read() will return the contents
            gzipped.seek(0)
            return gzipped
        else:
            return data_or_file

    def close_file(self):
        """Close any file that this attachment info has opened.
        """
        if self._my_file is not None:
            self._my_file.close()
            self._my_file = None


class JobResponse(object):
    """ The response for a job execution request.

    Attributes:
        jobid: The job id.
        execution_status: The execution status of the job.
        solution: The solution of the job, if it has been read. The solution
            is the output attachment which name is ``solution.ext``, where
            ext is the extension for the output result format ('TEXT', 'XML',
            'JSON', 'XLSX')
        job_info: A ``dict`` containing information on the job.
    """
    def __init__(self, jobid, executionStatus=None):
        self.jobid = jobid
        self.execution_status = executionStatus
        self.solution = None
        self.job_info = None


class DOcloudException(Exception):
    """ The base class for exceptions raised by the DOcplexcloud client."""
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class JobUploadError(DOcloudException):
    """ The error raised when an upload operation failed."""
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class ParameterError(DOcloudException):
    """ The error raised when a wrong parameter was passed to the API."""
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class DOcloudForbiddenError(DOcloudException):
    """ The error raised when the service cannot be used because of
    autorisation issues.
    """
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class DOcloudNotFoundError(DOcloudException):
    """ The error raised when an operation was requested on a non-existent
    resource.
    """
    def __init__(self, msg, url=None):
        message = msg if url is None \
            else "url={url}, message={msg}".format(url=url, msg=msg)
        Exception.__init__(self, message)
        self.message = message


class DOcloudInterruptedException(DOcloudException):
    """ The exception raised when an operation timed out.

    Attributes:
        message: The error message.
        jobid: The id of the job.
    """
    def __init__(self, msg, jobid=None):
        Exception.__init__(self, msg)
        self.message = msg
        self.jobid = jobid


class JobClient(object):
    """A client to create, submit and monitor jobs on DOcplexcloud.

    Examples:

        Submits a model as `.mod`, with data as a `.dat` file, then after
        the solve, the results is downloaded and saved as `results.json`::

            >>> client = JobClient(url, api_key)
            >>> resp = client.execute(input=["models/truck.dat",
                                             "models/truck.mod"],
                                    output="results.json")

        Submits a model which definition is first read in memory, then
        download results as `results.json`. Additionnaly, solve log is
        downloaded as `solver.log`. Data is sent gzipped. Wait for the job
        for 300 seconds. Once the solve is finished or if the wait time is
        elapsed, the job is deleted::

            >>> client = JobClient(url, api_key)
            >>> with open("models/truck.mod", "rb") as modFile:
            >>>     resp = client.execute(input=[{"name":"truck.mod",
                                                  "file":modFile},
                                                  "models/truck.dat"],
                                          output="results.json",
                                          log="solver.log",
                                          gzip=True,
                                          waittime=300,
                                          delete_on_completion=True)

    Attributes:
        url(str): The URL this client connects to.
        timeout(float): The timeout used when listening on sockets. The timeout
            is specified in seconds.
        verify(bool): Verifies SSL certificates for HTTPS requests.
        logger(Logger): A logger you can set if you want more verbose information.
        nice(float): When greater than zero, specifies a time in seconds between
            requests in loops.
    """
    DEFAULT_TIMEOUT = 180  # default timeout
    DEFAULT_ENCODING = "utf-8"  # default encoding used
    DEFAULT_NICE = 2  # default is to wait 2 sec between polls while waiting

    BOOLEAN_VALUES_FOR_URL = {True: 'true', False: 'false'}

    def __init__(self, url, api_key, client_secret=None,
                 proxies=None, max_retries=1):
        """ Initialize a JobClient.

        Args:
            url (str): The URL to connect to.
            api_key (str): The API key.
            client_secret (str): The client secret of the API key.
            proxies (dict): Optional dictionary mapping protocol to the URL of
               the proxy.
        """
        self.url = url
        self.api_key = api_key
        self.client_secret = client_secret
        # Create session
        self.session = requests.Session()
        # mount custom adapters for http and https for this session
        self.session.mount("http://",
                           requests.adapters.HTTPAdapter(max_retries=max_retries))
        self.session.mount("https://",
                           requests.adapters.HTTPAdapter(max_retries=max_retries))
        self.session.headers.update({'X-IBM-Client-Id': api_key})
        if self.client_secret is not None:
            self.session.headers.update({'X-IBM-Client-Secret': client_secret})
        # headers used for control messages
        self._base_headers = {}
        self._base_headers['Content-Type'] = 'application/json'
        # headers used to send streams
        self._stream_headers = {}
        self._stream_headers['Content-Type'] = 'application/octet-stream'
        # headers used to send gzipped streams
        self._gz_stream_headers = self._stream_headers.copy()
        self._gz_stream_headers['Content-Encoding'] = 'gzip'
        # change that value to change timeout
        self.timeout = JobClient.DEFAULT_TIMEOUT
        # set to a not None value if you want logging
        self.logger = None
        # verify SSL ?
        self.verify = True  
        self.nice = JobClient.DEFAULT_NICE
        # a callback that is called before a request is performed
        # called with parameters (rest method as String, url, *args, **kwargs)
        self.rest_callback = None
        # optional parameters for all requests
        self.requests_options = dict()
        if proxies is not None:
            self.requests_options['proxies'] = proxies
        self.retry_on_status = {502}  # on 502, retry
        self.retry_count = 3  # retry 3 times
        self.retry_wait = 2  # wait time between retries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _info(self, *args, **kwargs):
        if (self.logger is not None):
            self.logger.info(*args, **kwargs)

    def _check_status(self, response, ok_status):
        """ Checks that the request is performed correctly. """
        if not (response.status_code in ok_status):
            try:
                j = response.json()
                if "message" in j:
                    message = j["message"]
                elif "httpMessage" in j:
                    http_message = j["httpMessage"]
                    more_info = j.get("moreInformation", "")
                    if more_info != "":
                        more_info = "({0})".format(more_info)
                    http_code = j.get("httpCode", "")
                    message = "{0} {1} {2}".format(http_code, http_message,
                                                   more_info)
                else:
                    message = str(j)
            except ValueError:
                message = response.content
            if response.status_code == 403:
                raise DOcloudForbiddenError(message)
            elif response.status_code == 404:
                raise DOcloudNotFoundError(message)
            raise DOcloudException('{0}: {1}'.format(response.status_code,
                                                     message))

    def _check_ok(self, response):
        """ Checks that the request is performed correctly. """
        return self._check_status(response, [200])

    def _check_created(self, response):
        """ Checks that the request is performed correctly. """
        return self._check_status(response, [201])

    def _check_ok_no_content(self, response):
        """ Checks if the response is 204 No Content."""
        return self._check_status(response, [204])

    def _request(self, method, url, timeout=None, *args, **kwargs):
        tm_sec = self.timeout if timeout is None else timeout
        retry_count = self.retry_count
        response = None
        while retry_count > 0:
            if self.rest_callback:
                self.rest_callback(method, url,
                                   timeout=tm_sec,
                                   *args,
                                   **kwargs)
            response = self.session.request(method, url,
                                            timeout=tm_sec,
                                            *args,
                                            **kwargs)
            if response.status_code in self.retry_on_status:
                retry_count = retry_count - 1
                if self.retry_wait:
                    time.sleep(self.retry_wait)
            else:
                return response
        return response

    def _post(self, url, data=None, timeout=None):
        return self._request('POST', url,
                             data=data,
                             headers=self._base_headers,
                             verify=self.verify,
                             timeout=timeout,
                             **self.requests_options)

    def _get(self, url, timeout=None, stream=False):
        return self._request('GET',
                             url,
                             headers=self._base_headers,
                             verify=self.verify,
                             timeout=timeout,
                             stream=stream,
                             **self.requests_options)

    def _delete(self, url, timeout=None):
        return self._request('DELETE', url,
                             headers=self._base_headers,
                             verify=self.verify,
                             timeout=timeout,
                             **self.requests_options)

    def _put(self, url, data=None, timeout=None, gzip=False):
        headers = self._stream_headers
        if gzip:
            headers = self._gz_stream_headers
        # do not use self._base_headers here, but _stream_headers since
        # self._base_headers defines Content-Type: application/json
        return self._request('PUT', url,
                             data=data,
                             headers=headers,
                             verify=self.verify,
                             timeout=timeout,
                             **self.requests_options)

    def close(self):
        """ Closes this client and free up used resources.
        """
        self.session.close()

    def submit(self, input=None, timeout=None, gzip=False, parameters=None):
        """ Submits a job.

        Submits a job but does not wait for the execution to end.
        Attachments are uploaded prior to the execution.

        Args:
            input: List of attachments. Each attachment is a ``dict``
                specifying attachment name and attachment data, file or
                filename.
            timeout: The timeout for requests.
            parameters: A ``dict`` with additional job parameters.
                See `Job Parameters <https://developer.ibm.com/docloud/documentation/docloud/job-parameters/>`_.

        Returns:
            The id of the job created.
        """
        jobid = self.create_job(input=input,
                                gzip=gzip,
                                timeout=timeout,
                                parameters=parameters)
        # run model
        self.execute_job(jobid, timeout=timeout)

        return jobid

    def prepare_results_format_parameter_from_output_spec(self,
                                                          output=None,
                                                          parameters=None):
        # If ``output`` specification needs a result format parameter change,
        #    return the new parameters.
        #    Else return ``parameters``
        if output and isinstance(output, string_types):
            o = output
            extension = os.path.splitext(o)[1][1:].upper()
            is_ext_supported = extension in ['TEXT', 'TXT', 'XML', 'JSON', 'XLSX']
            is_resultsFormat_defined = parameters is not None and 'oaas.resultsFormat' in parameters
            # if output format was not specified in oaas.resultFormat,
            # set it depending on extension of output file
            if is_ext_supported and not is_resultsFormat_defined:
                if parameters is None:
                    parameters = {}
                else:
                    parameters = parameters.copy()
                parameters['oaas.resultsFormat'] = extension
                if extension == 'TXT':
                    parameters['oaas.resultsFormat'] = 'TEXT'
        return parameters

    def execute(self, input=None, output=None, load_solution=False, log=None,
                delete_on_completion=True, timeout=None, waittime=-1,
                gzip=False, parameters=None, continuous_logs=True,
                info_cb=None):
        """Submit and monitor a job execution.

        Submits a job and waits for the execution to end.
        Attachments are uploaded prior to the execution.
        After the execution has ended, job outputs and logs can be
        automatically downloaded.

        ``output`` can be:
            - a string representing a filename: after solve, the output
                attachment of name ``solution.extension`` is saved to that
                filename. ``extension`` is the extension of the filename.
                Extension must be one of the supported extensions of the solve
                engine ('TXT' if oaas.outputFormat is 'TEXT', 'XML', 'JSON',
                'XLSX').

                Example:

                    ``output="output.xlsx"`` forces the result format to be
                    xlsx and download ``solution.xlsx`` output attachment
                    after the solve.

            - a dict mapping attachment name to filenames.

                Example::

                    output= { 'output.json': 'file_output.json',
                              'output.csv' : 'file_output.csv' }

                After the solve, download output attachment of name
                ``output.json`` to local file ``file_output.json`` and
                ``output.csv`` to local file ``file_output.csv``

        Args:
            input: List of attachments. Each attachment is a ``dict`` specifying
                attachment name and attachment data, file or filename.
            timeout: The timeout for requests.
            output: The filename where to save job output or a mapping between
                output attachment names and local filenames to download
                attachments.
            load_solution: If True, the solution is loaded from the service
                once the solve has finished and returned in the job response
                object. If multiple output attachments are
                available, the ``solution`` field of the response contains
                the output attachment which name is ``solution.extension`` where
                ``extension`` is the output format extension.
            log: The filename where to save job logs or a file-like object to
                write logs to.
            continuous_logs: If True, logs are downloaded as available. If False,
                logs are downloaded and written to the log stream at job
                completion.
            delete_on_completion: If set, the job is deleted after it is
                completed.
            waittime: The maximum time to wait.
            gzip: If True, the input data is compressed using gzip before
                it is uploaded.
            parameters: A ``dict`` with additional job parameters.
                See `Job Parameters <https://developer.ibm.com/docloud/documentation/docloud/job-parameters/>`_.
            info_cb: A callback called each time a job info has been received
                from the server. The info_cb first parameter is a dict with
                the job info, see :meth:`~docloud.job.JobClient.get_job`.
        Returns:
            A ``JobResponse`` with the execution status of the job.

        Raises:
            DOcloudInterruptedException: if the maximum time to wait
                has elapsed and the job has not solved yet.
        """
        output_solution_filename = None
        if output and isinstance(output, string_types):
            output_solution_filename = output
        # check output format
        parameters = self.prepare_results_format_parameter_from_output_spec(output=output,
                                                                            parameters=parameters)
        solution_att_name = default_solution_name(parameters)

        # Create output mapping
        output_mapping = {}
        if output and isinstance(output, string_types):
            # save solution_att_name as output_solution_filename
            output_mapping[solution_att_name] = output_solution_filename
        if isinstance(output, dict):
            # assume output maps (attribute name -> filename)
            output_mapping = output
        # submit job
        jobid = self.submit(input=input, timeout=timeout, gzip=gzip,
                            parameters=parameters)
        response = None
        completed = False
        try:
            # monitor job execution
            status = self.wait_for_completion(jobid,
                                              timeout=timeout,
                                              waittime=waittime,
                                              log=log,
                                              continuous_logs=continuous_logs,
                                              info_cb=info_cb)
            # if waittime or timeout elapsed without this finishing,
            # an interruption exception should have been raised at this point.
            # it's safe to assume that the job has completed when we reach
            # this point.
            completed = True

            # prepare return value
            response = JobResponse(jobid, executionStatus=status)
            # get some job info
            response.job_info = self.get_job(jobid)
            attachments = response.job_info['attachments']
            has_solution = False
            solution = None
            for a in attachments:
                if a["type"] == "OUTPUT_ATTACHMENT":
                    output_data = None # downloaded output data for this attachment
                    if a["name"] in output_mapping:
                        output_data = self.download_job_attachment(jobid,
                                                                   a["name"])
                        with open(output_mapping[a["name"]], "wb") as f:
                            f.write(output_data)
                    if a["name"] == solution_att_name:
                        if a["name"] not in output_mapping:
                            # download data if it has not been downloaded
                            output_data = self.download_job_attachment(jobid,
                                                                       a["name"])
                        has_solution = True
                        solution = output_data
            if load_solution:
                response.solution = solution
        finally:
            # delete on completion if completed
            if delete_on_completion and completed:
                self.delete_job(jobid, timeout=timeout)
        return response

    def create_job(self, **kwargs):
        """ Creates a new job.

        The job parameters parameter is a dict containing various job and system
        parameters. See `Job Parameters <https://developer.ibm.com/docloud/documentation/docloud/job-parameters/>`_.

        Attachment definitions are ``dict`` containing info. Such ``dict`` can
        include the following key/values pair:

        - name (string): The name of the attachment.
        - length (int): The length of the attachment.

        Example:

            Creates a job with one attachment called 'diet.lp'

            >>> jobid = client.create_job(attachments=[{'name' : 'diet.lp'}])

        Args:
            applicationId (optional): The application id.
            applicationVersion (optional): The application version.
            input (optional): List of attachments. Each attachment is a
                ``dict`` specifying attachment name and attachment data, file
                or filename. When ``input`` is specified, any ``attachments``
                args are overridden. Attachments specified by ``input`` are
                automatically uploaded at job creation time.
            parameters (optional): A ``dict`` containing job parameters.
                See `Job Parameters <https://developer.ibm.com/docloud/documentation/docloud/job-parameters/>`_.
            attachments (optional): A list of ``dict`` containing attachment
                definitions.
            clientName (optional): The name of the client.

        Note:
            The ``kwargs`` are JSON encoded and passed to the DOcplexcloud service.

        Returns:
            The jobid.
        """
        input = kwargs.pop('input', None)
        gzip = kwargs.pop('gzip', False)
        timeout = kwargs.pop('timeout', None)
        # copy kwargs, trimming items which value is None.
        # This is needed since DOcplexcloud does not like entries like
        # "parameters" : null
        mykwargs = {}
        for key, value in iteritems(kwargs):
            if value is not None:
                mykwargs[key] = value

        # create attachament info if needed
        attachmentInfo = []
        if input:
            for raw_inp in input:
                attachmentInfo.append(JobAttachmentInfo(raw_inp))
            attachments = [{'name': a.name} for a in attachmentInfo]
            mykwargs['attachments'] = attachments

        # Create the job
        response = self._post(self.url + "/jobs",
                              data=json.dumps(mykwargs),
                              timeout=timeout)
        self._check_created(response)
        job_url = response.headers['location']
        job_id = job_url.rsplit("/", 1)[1]
        self._info(_("Created job, id = {job_id}").format(job_id=job_id))

        # upload attachments
        if attachmentInfo:
            for inp in attachmentInfo:
                self.upload_job_attachment(job_id, 
                                           attid=inp.name,
                                           data=inp.get_data_or_file(),
                                           gzip=gzip)
        return job_id

    def copy_job(self, jobid, shallow=None, **kwargs):
        """Creates a new job by copying an existing one.

        The existing job must not be running or waiting for execution. All
        creation data is copied over to the new job.  By default, the input
        attachments and their contents are copied in the new job. If a shallow
        copy is requested the new attachment will point to the existing job,
        and if it is deleted, accessing the attachment will raise an exception.
        Output attachments are not copied. Optionally, a job creation data can
        be passed to override the parameters and declare additional or
        replacement input attachments.

        Args:
            jobid: The id of the job to copy.
            shallow: Indicates if the copy is shallow.

        Note:
            The ``kwargs`` are JSON encoded and passed to the DOcplexcloud service as
            job creation data override.

        Returns:
            The id of the copy.
        """
        override_data = json.dumps(kwargs)
        shallow_option = ""
        if shallow is not None:
            shallow_option = "?shallow={0!s}".format(JobClient.BOOLEAN_VALUES_FOR_URL[shallow])

        url = '{base_url}/jobs/{jobid}/copy{shallow}'.format(base_url=self.url,
                                                             jobid=jobid,
                                                             shallow=shallow_option)
        response = self._post(url, data=override_data)
        self._check_created(response)
        job_url = response.headers['location']
        job_id = job_url.rsplit("/",1)[1]
        self._info(_("Copied job, id = {job_id}").format(job_id=job_id))
        return job_id

    def recreate_job(self, jobid, execute=None, **kwargs):
        """Creates a new job by replacing an existing one.

        This can be used to resubmit a failed job for example. The existing
        job must not be running or waiting for execution. All creation data
        is copied over to the new job. Input attachments are declared in the
        new job and the content is owned by the new job. Output attachments are
        ignored. Optionally, a job creation data can be passed to override the
        parameters and declare additional input attachments. The existing job
        is automatically deleted.

        Args:
            jobid: The id of the job to recreate.
            execute: Indicates if the job must be executed immediately.

        Note:
            The ``kwargs`` are JSON encoded and passed to the DOcplexcloud service as
            job creation data override.

        Returns:
            The id of the copy.
        """
        override_data = json.dumps(kwargs)
        execute_option = ""
        if execute is not None:
            execute_option = "?execute={0!s}".format(JobClient.BOOLEAN_VALUES_FOR_URL[execute])

        url = '{base_url}/jobs/{jobid}/recreate{execute}'.format(base_url=self.url,
                                                                 jobid=jobid,
                                                                 execute=execute_option)
        self._info(_("Recreating job id = {job_id}").format(job_id=jobid))
        response = self._post(url, data=override_data)
        self._check_created(response)
        job_url = response.headers['location']
        job_id = job_url.rsplit("/",1)[1]
        self._info(_("Recreated job, new id = {job_id}").format(job_id=job_id))
        return job_id

    def abort_job(self, jobid, timeout=None):
        """ Aborts the specified job.

        Args:
             jobid: The id of the job.
            timeout: The timeout for requests.
        """
        url = '{base_url}/jobs/{jobid}/execute'.format(base_url=self.url,
                                                       jobid=jobid)
        response = self._delete(url, timeout=timeout)
        self._check_ok_no_content(response)

    def create_job_attachment(self,
                              jobid,
                              attachment_creation_data,
                              timeout=None):
        """ Creates an attachment for the job which id is specified.

        Attachment creation data is a ``dict`` which can contain the following keys:

        - name: The name of the attachment.
        - length: The length of the attachment. (Optional)

        Args:
            jobid: The id of the job.
            attachment_creation_data: Attachment creation data
            timeout: The timeout for requests. 

        Returns:
            The attribute id.
        """
        url = '{base_url}/jobs/{jobid}/attachments'.format(base_url=self.url,
                                                           jobid=jobid)
        response = self._post(url,
                              data=json.dumps(attachment_creation_data),
                              timeout=timeout)
        self._check_created(response)
        attachment_url = response.headers['location']
        attid = attachment_url.rsplit("/", 1)[1]
        return attid

    def delete_all_jobs(self, timeout=None):
        """ Deletes all jobs for the user.

        Args:
            timeout: The timeout for requests.
        """
        url = '{base_url}/jobs'.format(base_url=self.url)
        self._info(_("deleting all jobs"))
        response = self._delete(url, timeout=timeout)
        self._check_ok_no_content(response)


    def delete_job(self, jobid, timeout=None):
        """ Deletes the specified job.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.

        Returns:
            True if the job deletion was successful.
        """
        url = '{base_url}/jobs/{jobid}'.format(base_url=self.url, jobid=jobid)
        self._info(_("deleting job {job_id}").format(job_id=jobid))
        response = self._delete(url, timeout=timeout)
        self._check_ok(response)
        j = response.json()
        deleteStatus = (j['status'] == 'DELETED')
        self._info(_("   deleted with status = {}").format(j['status']))
        return deleteStatus

    def delete_job_attachment(self, jobid, attid, timeout=None):
        """ Deletes the specified job attachment.

        Args:
            jobid: The id of the job.
            attid: The attachment id.
            timeout: The timeout for requests

        Returns:
             True if the job attachment deletion was successful.
        """
        url = '{base_url}/jobs/{jobid}/attachments/{attid}'.format(
              base_url=self.url, jobid=jobid, attid=attid)
        self._info(_("deleting job attachment {attid} of job {jobid}")
                   .format(attid=attid, jobid=jobid))
        response = self._delete(url, timeout=timeout)
        self._check_ok(response)
        j = response.json()
        deleteStatus = (j['status'] == 'DELETED')
        self._info(_("   deleted with status = {}").format(j['status']))
        return deleteStatus

    def delete_job_attachments(self, jobid, timeout=None):
        """ Deletes all job attachment.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.
        """
        url = '{base_url}/jobs/{jobid}/attachments'.format(base_url=self.url,
                                                           jobid=jobid)
        self._info(_("deleting all job attachments of job {}").format(jobid))
        response = self._delete(url, timeout=timeout)
        self._check_ok_no_content(response)

    def download_job_attachment(self, jobid, attid, timeout=None):
        """ Download the specified job attachment.

        The attachment data is fully loaded into memory then returned.

        Args:
            jobid: The id of the job.
            attid: The attachment id.
            timeout: The timeout for requests.

        Returns:
            The contents of the attachment as a byte array.
        """
        url = '{base_url}/jobs/{jobid}/attachments/{attid}/blob'\
            .format(base_url=self.url, jobid=jobid, attid=attid)
        self._info(_("downloading attachment {attid} of job {jobid}")
                   .format(attid=attid, jobid=jobid))
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.content

    def download_job_attachment_as_stream(self, jobid, attid, timeout=None):
        """ Download the specified job attachment as stream.

        Instead of downloading the data, the response object resulting from
        the request is returned.

        Example:
            This is an example code downloading the attachment to a local
            file::

                r = client.download_job_attachment_as_stream(jobid, attid)
                with open('local_filename', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                    f.flush()

        Args:
            jobid: The id of the job.
            attid: The attachment id.
            timeout: The timeout for requests.  

        Returns:
            The reponse object to iter from.
        """
        url = '{base_url}/jobs/{jobid}/attachments/{attid}/blob'\
            .format(base_url=self.url, jobid=jobid, attid=attid)
        self._info(_("downloading attachment {attid} of job {jobid} as stream")
                   .format(attid=attid, jobid=jobid))
        response = self._get(url, timeout=timeout, stream=True)
        self._check_ok(response)
        return response

    def download_job_attachment_as_df(self, jobid, attid, timeout=None):
        """ Download the specified job attachment as a `pandas.DataFrame`.

        The attachment must be a valid `.csv` file.

        Example:
            This is an example code downloading the attachment to a local
            file::

                df = client.download_job_attachment_as_df(jobid, 'output_df.csv')

        Args:
            jobid: The id of the job.
            attid: The attachment id.
            timeout: The timeout for requests.  

        Returns:
            The data frame.

        Raises:
            NotImplementedError if `pandas` is not available
        """
        if pandas is None:
            raise NotImplementedError(_("Pandas must be installed for download_job_attachment_as_df()"))
        data = self.download_job_attachment(jobid, attid=attid, timeout=timeout)
        result_df = pandas.read_csv(BytesIO(data), index_col=None)
        return result_df

    def download_job_log(self, jobid, timeout=None):
        """ Downloads the job logs and returns the contents as a string.

        Args:  
            jobid: The id of the job.
            timeout: The timeout for requests.  

        Returns:
            The contents of the logs as a string.
        """
        url = '{base_url}/jobs/{jobid}/log/blob'.format(base_url=self.url,
                                                        jobid=jobid)
        self._info(_("downloading log of job {jobid}").format(jobid=jobid))
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.content

    def execute_job(self, jobid, timeout=None):
        """ Submits the specified job for execution.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.
        """
        url = '{base_url}/jobs/{jobid}/execute'.format(base_url=self.url,
                                                       jobid=jobid)
        self._info(_("executing job {}").format(jobid))
        response = self._post(url, timeout=timeout)
        self._check_ok_no_content(response)

    def get_failure_info(self, jobid, timeout=None):
        """ Gets the failure information for the specified job.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.

        Returns:
            The job information as a dict containing the failure
            information.
        """
        url = '{base_url}/jobs/{jobid}/failure'.format(base_url=self.url,
                                                       jobid=jobid)
        self._info(_("getting job failure info for job {}").format(jobid))
        response = self._get(url, timeout=timeout)
        self._check_status(response, [200, 204])
        if (response.status_code == 200):
            return response.json()
        return None

    def get_job(self, jobid, timeout=None):
        """ Gets the information for the specified job.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.

        Returns:
            The job information as a ``dict`` containing the job information.
            See `DOcplexcloud REST API reference <https://api-swagger-oaas.docloud.ibmcloud.com/api_swagger/?cm_mc_uid=15669745777015105815231&cm_mc_sid_50200000=1512381269#!/jobs/getJob>`_.
        """
        url = '{base_url}/jobs/{jobid}'.format(base_url=self.url, jobid=jobid)
        self._info(_("getting job info for job {}").format(jobid))
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.json()

    def get_all_jobs(self, timeout=None):
        """ Returns all jobs for this user.

        Args:
            timeout: The timeout for requests.

        Returns:
            A list of job information. Job information is a ``dict``
            containing the job information.
        """
        url = '{base_url}/jobs'.format(base_url=self.url)
        self._info(_("getting all jobs info"))
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.json()

    def get_job_attachment(self, jobid, attid, timeout=None):
        """ Returns the job attachment info for the specified job.

        Args:
            jobid: The id of the job.
            attid: The attachment id.
            timeout: The timeout for requests.  

        Returns:
            The job attachment info as a ``dict``.
        """
        url = '{base_url}/jobs/{jobid}/attachments/{attid}'\
            .format(base_url=self.url, jobid=jobid, attid=attid)
        self._info(_("getting attachment info {attid} from job {jobid}")
                   .format(attid=attid, jobid=jobid))
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.json()

    def get_job_attachments(self, jobid, timeout=None):
        """ Returns the attachments for a given job.

        The returned list of attachments is a list of ``dict``.

        Each entry can contain the following keys:

        - name: The name of the attachment.
        - type: The type of the attachment.
        - length: The length of the attachement.

        Args:
            jobid: The id of the job.
            attid: The attachment id.

        Returns:
            A list of the attachments.
        """
        url = '{base_url}/jobs/{jobid}/attachments'.format(base_url=self.url,
                                                           jobid=jobid)
        self._info(_("getting all attachment info from job {}").format(jobid))
        response = self._get(url, timeout=None)
        self._check_ok(response)
        return response.json()

    def get_execution_status(self, jobid, timeout=None):
        """ Returns the execution status of job.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.

        Returns:
            A ``JobExecutionStatus``
        """
        url = '{base_url}/jobs/{jobid}/execute'.format(base_url=self.url,
                                                       jobid=jobid)
        # no logging here, since this can be used in long running loops
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        j = response.json()
        return JobExecutionStatus[j["executionStatus"]]

    def get_log_items(self, jobid, start=None, continuous=None, timeout=None):
        """ Returns a list of log items for a job.

        Args:
            jobid: The id of the job.
            start: The log item starting index.
            continuous: If True, return items as they become available.
            timeout: The timeout for requests.

        Returns:
            A list containing the log items.
        """
        params = []
        if (start is not None):
            params.append("start={0}".format(start))
        if (continuous is not None):
            params.append("continuous={0}".format(continuous))

        paramString = ""
        if (len(params) != 0):
            paramString = "?" + '&'.join(params)

        url = '{base_url}/jobs/{jobid}/log/items{p}'.format(base_url=self.url, 
                                                            jobid=jobid, 
                                                            p=paramString)
        response = self._get(url, timeout=timeout)
        self._check_ok(response)
        return response.json()

    def kill_job(self, jobid, timeout=None):
        """ Kills the specified job.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.
        """
        url = '{base_url}/jobs/{jobid}/execute?kill=true'\
            .format(base_url=self.url, jobid=jobid)
        self._info(_("Killing job {jobid}").format(jobid=jobid))
        response = self._delete(url, timeout=timeout)
        self._check_ok_no_content(response)

    def upload_job_attachment(self, jobid, attid,
                              file=None, data=None, filename=None,
                              timeout=None, gzip=False):
        """ Uploads the attachment ``attid`` for the specified job id. One of
        the following can be specified:

        - file : The file which contents is read using ``get()`` before sent.
        - data : The data as a bytearray or as a `pandas.DataFrame`
        - filename : The name of a file that is opened and ``get()``.

        Args:
            jobid: The id of the job.
            attid: The id of the attribute.
            file: If specified, the file or stream object to read data from.
            data: The data of the attachment as a byte array.
            filename: If specified, the name of the file to be read.
            timeout: The timeout for requests.
            gzip: If True, the data is gzipped before sent
        """
        att = JobAttachmentInfo({'name': attid},
                                file=file, data=data, filename=filename)
        try:
            # now let's upload that
            url = '{base_url}/jobs/{jobid}/attachments/{attid}/blob'\
                .format(base_url=self.url, jobid=jobid, attid=attid)
            self._info(_("uploading attachment to {url}").format(url=url))

            response = self._put(url,
                                 data=att.get_data_or_file(gzip=gzip),
                                 gzip=gzip)
            self._check_ok_no_content(response)
        finally:
            att.close_file()

    def wait_and_get_solution(self, jobid, timeout=None, waittime=-1,
                              nice=None):
        """Waits for the specfied job to finish, then download solution.

        This method calls ``wait_for_completion()`` then download solution
        if any.

        This will wait for ``waittime`` seconds. A ``waittime`` of -1 means
        waiting indefinitely.

        The ``timeout`` parameter controls the timeout to listen on sockets.

        If ``nice`` is specified, this method will wait for that amount of
        seconds between each execution status query.

        When the job has multiple output attachment, only the one which name
        is ``solution.ext`` (with ``ext`` in: 'TEXT', 'XML', 'JSON', 'XLSX')
        is downloaded.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.
            waittime: The maximum time to wait.
            nice: Additional sleep time between status requests.

        Returns:
            A ``JobResponse`` with the execution status of the job and the
                solution.
        """
        status = self.wait_for_completion(jobid,
                                          timeout=timeout,
                                          waittime=waittime,
                                          nice=nice)
        # prepare return value
        response = JobResponse(jobid, executionStatus=status)
        # get some job info
        response.job_info = self.get_job(jobid)
        parameters = response.job_info.get('parameters')
        solution_att_name = default_solution_name(parameters)
        attachments = response.job_info['attachments']
        has_solution = False
        for a in attachments:
            if a["type"] == "OUTPUT_ATTACHMENT" and a["name"] == solution_att_name:
                has_solution = True
        # download solution
        if status is not JobExecutionStatus.FAILED and has_solution:
            response.solution = self.download_job_attachment(jobid,
                                                             solution_att_name)
        return response

    def wait_for_completion(self, jobid, timeout=None,
                            waittime=-1, nice=None,
                            log=None, continuous_logs=True,
                            info_cb=None):
        """Waits for the specified job to finish.

        This loops and queries for the job execution status until the status is
        ended.

        This will wait for ``waittime`` seconds. A ``waittime`` of -1 means
        waiting indefinitely.

        The ``timeout`` parameter controls the timeout to listen on sockets.

        If ``nice`` is specified, this method will wait for that amount of
        seconds between each execution status query.

        Args:
            jobid: The id of the job.
            timeout: The timeout for requests.
            waittime: The maximum time to wait.
            nice: Additional sleep time between status requests.
            log: The filename where to save job logs or a file-like object to
                write logs to.
            continuous_logs: If True, logs are downloaded as available. If
                False, logs are downloaded and written to the log stream at job
                completion.
            info_cb: A callback called each time a job info has been received
                from the server. The info_cb first parameter is a dict with
                the job info, see :meth:`~docloud.job.JobClient.get_job`.
        Returns:
            The job execution status as a ``JobExecutionStatus``.
        Raises:
            DOcloudInterruptedException: if the ``waittime`` has expired.
        """
        nice_sec = self.nice if nice is None else nice

        last_seqid = 0
        log_file = None
        my_log_file = None  # when we open the file, remember to close() later
        if log:
            if isinstance(log, six.string_types):
                my_log_file = open(log, "w")
                log_file = my_log_file
            else:
                log_file = log  # use log as a file like

        time_limit = time.time() + waittime
        self._info(_("waiting for completion of {jobid} with a waittime of {waittime}").format(jobid=jobid,
                                                                                               waittime=waittime))
        status = None
        try:
            while (not JobExecutionStatus.isEnded(status)):
                info = self.get_job(jobid, timeout=timeout)
                if info_cb:
                    info_cb(info)
                status = JobExecutionStatus[info['executionStatus']]
                if (waittime >= 0) and (time.time() > time_limit):
                    raise DOcloudInterruptedException(_("Timeout after {0}").format(waittime), jobid=jobid)
                # if the status is ended, don't need to wait
                if (not JobExecutionStatus.isEnded(status)) and (nice_sec > 0):
                    time.sleep(nice_sec)  # sleep to be nice
                # down load logs if needed
                if log_file and continuous_logs:
                    last_seqid = download_and_format_log_items(self, jobid,
                                                               last_seqid,
                                                               log_file)
            # at the end of the loop, we want to download the last chunk of logs
            if log_file:
                last_seqid = download_and_format_log_items(self, jobid,
                                                           last_seqid,
                                                           log_file)
        finally:
            if my_log_file:
                my_log_file.close()
        return status
