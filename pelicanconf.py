#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Rajesh Arasada'
SITENAME = "Rajesh Arasada's Blog"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = False

OUTPUT_PATH = '../output'

THEME = 'theme'

PLUGIN_PATHS = ['plugins/', ]

PLUGINS = ['i18n_subsites', ]

JINJA_ENVIRONMENT = {
    'extensions': ['jinja2.ext.i18n'],
}

BOOTSTRAP_THEME = 'flatly'

PYGMENTS_STYLE = 'monokai'

LINKS = (('NYC Data Science Academy', 'https://nycdatascience.com/'),
     ('Wikipedia', 'https://wikipedia.org'),
     ('TensorFlow', 'https://www.tensorflow.org/'),
     ('stack overflow','https://stackoverflow.com/'),)

SOCIAL = (('LinkedIn', 'https://www.linkedin.com/in/rajesharasada/'),
         ('Github', 'https://github.com/RajeshArasada'),)

        
ARTICLE_PATHS = ['articles']

STATIC_PATHS = ['img', 'pdf']

  
PAGE_PATHS = ['pages']

# The ARTICLE_URL variable states what should display in the web browser's address bar while 
# The ARTICLE_SAVE_AS variable defines where the article being generated should be output to.

ARTICLE_URL = 'articles/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'articles/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'

PAGE_URL = 'pages/{slug}/'
PAGE_SAVE_AS = 'pages/{slug}/index.html'
CATEGORY_URL = 'category/{slug}'
CATEGORY_SAVE_AS = 'category/{slug}/index.html'
TAG_URL = 'tag/{slug}'
TAG_SAVE_AS = 'tag/{slug}/index.html'



# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True