# This file is NOT licensed under the GPLv3, which is the license for the rest
# of YouCompleteMe.
#
# Here's the license text for this file:
#
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

import os
import sys
import subprocess
import ycm_core

from collections import defaultdict

C_SOURCE_EXTENSIONS = ['.c']
CXX_SOURCE_EXTENSIONS = ['.cpp', '.cxx', '.cc']
SOURCE_EXTENSIONS = C_SOURCE_EXTENSIONS + CXX_SOURCE_EXTENSIONS
C_HEADER_EXTENSIONS = ['.h']
CXX_HEADER_EXTENSIONS = ['.hxx', '.hpp']
HEADER_EXTENSIONS = C_HEADER_EXTENSIONS + CXX_HEADER_EXTENSIONS


def MakeRelativePathsInFlagsAbsolute(flags, working_directory):
    if not working_directory:
        return list(flags)
    new_flags = []
    make_next_absolute = False
    path_flags = ['-isystem', '-I', '-iquote', '--sysroot=']
    for flag in flags:
        new_flag = flag

        if make_next_absolute:
            make_next_absolute = False
            if not flag.startswith('/'):
                new_flag = os.path.join(working_directory, flag)

        for path_flag in path_flags:
            if flag == path_flag:
                make_next_absolute = True
                break

            if flag.startswith(path_flag):
                path = flag[len(path_flag):]
                new_flag = path_flag + os.path.join(working_directory, path)
                break

        if new_flag:
            new_flags.append(new_flag)
    return new_flags


def IsHeaderFile(filename):
    extension = os.path.splitext(filename)[1]
    return extension in HEADER_EXTENSIONS


def ParseCMakeDependFile(cwd, dependfile, filename):
    f = open(dependfile, 'r')
    dependfiles = defaultdict(list)
    objfiles = []
    for line in f:
        if not line.lstrip().startswith('#'):
            line_split = line.split(':')
            if len(line_split) == 2:
                objfile = line_split[0]
                df_relative = line_split[1].strip(' \n')
                df = os.path.join(cwd, df_relative)
                df = os.path.abspath(df)
                if df == filename:
                    objfiles.append(objfile)
                elif os.path.splitext(df)[1] in SOURCE_EXTENSIONS:
                    dependfiles[objfile].append(df)
    for objfile in objfiles:
        for df in dependfiles[objfile]:
            yield df


def CMakeDependFilePath(cwd):
    for dirpath, dirnames, filenames in os.walk(cwd):
        for filename in filenames:
            if filename == 'depend.make':
                yield os.path.join(dirpath, filename)


def GuessCompilationInfoForHeader(cwd, database, filename):
    for dependfile in CMakeDependFilePath(cwd):
        for replacement_file in ParseCMakeDependFile(cwd,
                dependfile, filename):
            compilation_info = database.GetCompilationInfoForFile(
                    replacement_file)
            if compilation_info.compiler_flags_:
                return (compilation_info, replacement_file)

    basename = os.path.splitext(filename)[0]
    for extension in SOURCE_EXTENSIONS:
        replacement_file = basename + extension
        if os.path.isfile(replacement_file):
            compilation_info = database.GetCompilationInfoForFile(
                    replacement_file)
            if compilation_info.compiler_flags_:
                return (compilation_info, replacement_file)
    return (None, None)


def GetCompilationInfoForFile(cwd, filename):
    database = ycm_core.CompilationDatabase(cwd)
    if database:
        if IsHeaderFile(filename):
            compilation_info, src_file = GuessCompilationInfoForHeader(
                    cwd, database, filename)
            if src_file:
                filename = src_file
        else:
            compilation_info = database.GetCompilationInfoForFile(filename)

        if compilation_info:
            final_flags = MakeRelativePathsInFlagsAbsolute(
                    compilation_info.compiler_flags_,
                    compilation_info.compiler_working_dir_)
            return (filename, final_flags)
    return (filename, [])


def KernelFlags(filename, flags):
    flags.append("-UCC_HAVE_ASM_GOTO")

    # remove flags that do not work with clang
    to_remove = [
            {'-mno-80387'},
            {'-mno-fp-ret-in-387'},
            {'-maccumulate-outgoing-args'},
            {'-fno-delete-null-pointer-checks'},
            {'-fno-var-tracking-assignments'},
            {'-mfentry'},
            {'-fconserve-stack'},
            ]
    for flag in to_remove:
        try:
            flags.remove(flag)
        except ValueError:
            pass


def SourceLang(filename, database):
    ext = os.path.splitext(filename)[-1]
    lang = []

    # is it a C or C++ file?
    if ext in C_SOURCE_EXTENSIONS:
        lang.append("-x")
        lang.append("c")
        if not database:
            lang.append("-std=c11")
    else:
        lang.append("-x")
        lang.append("c++")
        if not database:
            lang.append("-std=c++11")

    return lang


def DefaultIncludes(filename, flags):

    f = open('/dev/null', 'rw')
    proc = subprocess.Popen(
            ["clang", "-v", "-E"] + SourceLang(filename, False) + ["-"],
            stdin=f, stderr=subprocess.PIPE,
            stdout=f)

    is_include_path = False
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        #line = line.decode("utf-8")
        if line.startswith("#include"):
            is_include_path = True
        elif is_include_path and line.startswith(' '):
            flags.append("-isystem")
            flags.append(line[1:-1])
        else:
            is_include_path = False

    f.close()


def FlagsForFile(filename, **kwargs):
    cwd = ""
    try:
        cwd = str(kwargs['client_data']['getcwd()'])
    except:
        pass
    filename, final_flags = GetCompilationInfoForFile(cwd, filename)
    final_flags.extend(SourceLang(filename, final_flags))
    if os.path.isfile(os.path.join(cwd, 'Kbuild')):
        KernelFlags(filename, final_flags)
    elif '-nostdinc' not in final_flags:
        # this is only needed for filename completion
        DefaultIncludes(filename, final_flags)

    # NOTE: This is just for YouCompleteMe; it's highly likely that your project
    # does NOT need to remove the stdlib flag. DO NOT USE THIS IN YOUR
    # ycm_extra_conf IF YOU'RE NOT 100% SURE YOU NEED IT.
    try:
        final_flags.remove('-stdlib=libc++')
    except ValueError:
        pass

    return {
            'flags': final_flags,
            'do_cache': True
            }
