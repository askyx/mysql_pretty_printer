import sys
import gdb
import os
import os.path

if gdb.current_objfile () is not None:
    path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, path)
  
from printer import register_mysql_printers
register_mysql_printers(gdb.current_objfile())