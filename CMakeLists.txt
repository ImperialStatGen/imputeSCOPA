project(scopa)
cmake_minimum_required(VERSION 2.0)

set(Ranger_Flags "")
set(Ranger_DIR $ENV{RANGER_DIR})
set(Ranger_SRCDIR ${Ranger_DIR}/src)
if (EXISTS ${Ranger_DIR}/local)
  set(Ranger_Flags "-DSID_VECTOR")
  set(Ranger_SRCDIR ${Ranger_DIR}/)
endif()
## ======================================================================================##
## Check for C++11. For GCC this is >=4.7
## ======================================================================================##
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  message("Compiler with C++11 support found.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler, i.e. gcc >= 4.7 or Clang >= 3.0.")
endif()

## ======================================================================================##
## Compiler flags
## ======================================================================================##
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

## ======================================================================================##
## In Clang phtread flag only for compiler, not for linker. For
## windows use static linking
## ======================================================================================##
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS -pthread)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows") 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -static")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -fpic -g -fstack-protector-strong ${Ranger_Flags}")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -pthread -O2 -fpic")
endif()

## ======================================================================================##
## Subdirectories and source files
## ======================================================================================##
include_directories(${Ranger_SRCDIR} include/) 

file(GLOB_RECURSE RANGER_SOURCES ${Ranger_SRCDIR}/*.cpp)
FOREACH(item ${RANGER_SOURCES})
  IF(${item} MATCHES "main.cpp" OR ${item} MATCHES "RcppExports.cpp" OR ${item} MATCHES "rangerCpp.cpp" OR ${item} MATCHES "DataSparse.cpp" OR ${item} MATCHES "utilityRcpp.cpp")
    LIST(REMOVE_ITEM RANGER_SOURCES ${item})
  ENDIF(${item} MATCHES "main.cpp" OR ${item} MATCHES "RcppExports.cpp" OR ${item} MATCHES "rangerCpp.cpp" OR ${item} MATCHES "DataSparse.cpp" OR ${item} MATCHES "utilityRcpp.cpp")
ENDFOREACH(item)
file(GLOB SOURCES cpp/*.cpp)

## ======================================================================================##
## Debug and release targets
## ======================================================================================##
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )
ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

## ======================================================================================##
## Executable
## ======================================================================================##
add_executable(imputeSCOPA ${SOURCES} ${RANGER_SOURCES})

