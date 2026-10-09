"""All of Foreman's blueprints, registered by the app (and by tests)."""

from foreman.job import job
from foreman.job_list import job_list
from foreman.job_queue import job_queue
from foreman.job_tree import job_tree
from foreman.search import search
from foreman.settings_page import settings_page

BLUEPRINTS = [job, job_list, job_tree, job_queue, search, settings_page]
