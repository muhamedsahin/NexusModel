#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nexusdata::nexusdata" for configuration "Debug"
set_property(TARGET nexusdata::nexusdata APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(nexusdata::nexusdata PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/nexusdata.lib"
  )

list(APPEND _cmake_import_check_targets nexusdata::nexusdata )
list(APPEND _cmake_import_check_files_for_nexusdata::nexusdata "${_IMPORT_PREFIX}/lib/nexusdata.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
