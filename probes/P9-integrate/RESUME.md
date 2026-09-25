Master here. The laptop battery hit 0% at 00:31 and it hibernated; it rebooted at 01:41 (now on the charger). Every
process died: detached jobs, GPU lock holders, everything. Check which outputs on disk are complete (files written
mid-crash may be truncated — verify they open and row counts are right) and redo only what is missing. Continue your
task from where you were. Rules unchanged: never end your turn to wait; detach long jobs; bounded polling.
