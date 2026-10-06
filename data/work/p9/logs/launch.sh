#!/usr/bin/env bash
# usage: launch.sh <name> <cmd...> -> nohup-detached; log data/work/p9/logs/<name>.log ("EXIT n" last), pid <name>.pid
name=$1; shift
cd /home/harsh1/github-repos/amc-26-wt/p9-integrate
L=data/work/p9/logs
nohup setsid bash -c '"$@" > '"$L/$name"'.log 2>&1; echo "EXIT $?" >> '"$L/$name"'.log' _ "$@" < /dev/null > /dev/null 2>&1 &
echo $! > $L/$name.pid
