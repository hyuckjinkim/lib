#!/bin/bash
# Example: lib/delete_macos_dotfiles.sh

# set root_dir and pattern
root_dir='/d/Github/hyuckjinkim'
pattern='._*'

# start time
start_time=$(date +"%Y/%m/%d %H:%M:%S")
start_seconds=$(date +%s)
echo "> Start Time : $start_time"

# count files to delete
file_count=$(find "$root_dir" -name "$pattern" | wc -l)
file_count_formatted=$(printf "%'d\n" $file_count)
echo "> Files to delete: $file_count_formatted"

# progress showing function
show_progress() {
    local delay=0.03
    local spinstr='|/-\'
    local pbar
    local i
    printf "> Delete Files :   "
    while true; do
        for (( i=0; i<${#spinstr}; i++ )); do
            pbar=${spinstr:i:1}
            printf "\b\b%s\b" "[$pbar]"
            sleep $delay
        done
    done
}

# clean up when interrupted
cleanup() {
    kill $PROGRESS_PID 2>/dev/null
    wait $PROGRESS_PID 2>/dev/null
    printf "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r"
    echo "> Deletion interrupted."
    exit 1
}

# trap setting for cleanup function
trap cleanup SIGINT

# start progress showing function
show_progress &
PROGRESS_PID=$!

# delete the files startswith '._'
find "$root_dir" -name "$pattern" -delete

# stop progress showing function
kill $PROGRESS_PID
wait $PROGRESS_PID 2>/dev/null
printf "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r"

# end time
end_time=$(date +"%Y/%m/%d %H:%M:%S")
end_seconds=$(date +%s)
echo "> End Time : $end_time"

# elapsed time
elapsed_seconds=$((end_seconds - start_seconds))
elapsed_minutes=$(echo "scale=2; $elapsed_seconds / 60" | bc)
echo "> Elapsed Time: $elapsed_minutes minutes"

# completed
echo "> Deletion completed."