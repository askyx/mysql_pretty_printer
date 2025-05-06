
cmake_minimum_required(VERSION 3.10.0)
project(mysql_pretty_printer VERSION 1.0.0)

find_program(
  MYSQL_CONFIG mysql_config
  PATHS ENV PATH
  PATH_SUFFIXES bin)

if(NOT MYSQL_CONFIG)
  message(FATAL_ERROR "Could not find mysql_config")
else()
  set(MySQL_FOUND TRUE)
endif()

message(STATUS "Found mysql_config as ${MYSQL_CONFIG}")

execute_process(
  COMMAND ${MYSQL_CONFIG} --variable=pkglibdir
  OUTPUT_VARIABLE MYSQL_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# which command can get bin dir directly?
string(REPLACE "lib" "bin" MYSQL_BIN_DIR ${MYSQL_INCLUDE_DIR})

message(STATUS "Found MySQL bin directory as ${MYSQL_BIN_DIR}")

find_package(Python3 REQUIRED)

set(PTINTER ${CMAKE_SOURCE_DIR}/mysqld-gdb.py)
set(GDBINIT $ENV{HOME}/.gdbinit)


MESSAGE(STATUS "install printer to postgres path: ${MYSQL_BIN_DIR}")
install(FILES ${PTINTER} DESTINATION ${MYSQL_BIN_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/printer DESTINATION ${MYSQL_BIN_DIR})

MESSAGE(STATUS "append debugger init file to: ${GDBINIT}")
file(APPEND ${GDBINIT} "add-auto-load-safe-path ${MYSQL_BIN_DIR}\n")