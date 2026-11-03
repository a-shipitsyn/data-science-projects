# AC2 access controller — programming guide

## Overview
The AC2 is a standalone access controller for one door with up to 2,000 users.
It supports 125 kHz proximity cards and key fobs and has a keypad for PIN codes.

## Adding keys
1. Enter programming mode: enter the master code and press #.
   The default master code is 9999 (1234 on units produced before 2024).
2. Press 1 to add keys, then present each card or key fob to the reader.
3. Press * to leave programming mode.

## Deleting keys
Enter programming mode and press 2, then present the card to delete.
To delete all keys press 2, then enter 0000 and press #.

## Door open time
The relay time is set from 1 to 99 seconds, default 5 seconds:
enter programming mode, press 4, enter the time and press #.

## Power supply
12 V DC, maximum current 150 mA without the lock.
