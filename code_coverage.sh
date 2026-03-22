#!/bin/bash
# if you want to check the code coverage, follow these steps would help
# 1.cmake set -DCODE_COVERAGE=ON or set(C_CXX_FLAGS " -fprofile-arcs -ftest-coverage   ${C_CXX_FLAGS}" )
# 2.build gb
# 3.install gcovr, "/project/ai/scratch01/shedu01/software/test_tool/gcovr_offline/"
# 4.run your program
# run this script to generate code coverage report, it will take some time ,be patient
# this script accept two args
#    argv1 is the folder you want to get the code coverage report, multiple directories are used, separated by commas.if you set this, it will be faster
#    arfv2 is the name of report
#    if you want to get more than one specified dirs ,use like
#    ./code_coverage.sh  "src/tongue, src/vm"

#set -x
sub_dir=""
output_file="coverage.html"


if [ $# -ge 1 ]; then
    sub_dir="$1"
fi


if [ $# -ge 2 ]; then
    output_file="$2"
fi

# make report dir
if [ ! -d "code_coverage_report" ]; then
    mkdir code_coverage_report
fi


filter_args=()
if [ -z "$sub_dir" ]; then
  filter_args+=(--filter "\.\./src")
else
  IFS=',' read -ra dirs <<< "$sub_dir"
  for d in "${dirs[@]}"; do
      d_trimmed=$(echo "$d" | xargs)
      filter_args+=(--filter "\.\./$d_trimmed")
  done
fi

echo $filter_args


cd build || { echo "❌ build dir not exist"; exit 1; }


gcovr -r .. \
      --html \
      --html-nested \
      --merge-mode-functions=merge-use-line-max \
      "${filter_args[@]}" \
      --gcov-ignore-parse-errors \
      -o "../code_coverage_report/$output_file"