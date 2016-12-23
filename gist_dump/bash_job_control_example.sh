#! /bin/bash
check_jobs() {
    # 8 jobs, check to see if able to run every 3 seconds
    while [ $(jobs | wc -l) -ge 8 ]; do
        sleep 3;
    done
}

expensive_job() {
    for i in a b c; do
        # jobs of random length from 0 -> 6
        sleep $(( $RANDOM % 7 ))
    done
    echo "job complete"
}

while true; do
    check_jobs; expensive_job &
    echo "new job"
done