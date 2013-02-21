#!/usr/bin/awk -f

BEGIN {
    RED = "[31;1m";
    GREEN = "[32m";
    X = "[0m";
}

/^SUCCESS/ { print GREEN $0 X;  next; }
/^FAILURE/ { print RED $0 X; next; }
{ print  $0; }
