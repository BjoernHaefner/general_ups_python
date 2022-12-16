#!/bin/bash

script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Download synthetic data set joyfulyell hippie
wget -P $script_path "https://vision.in.tum.de/webshare/u/haefner/GeneralUPS/synthetic_joyfulyell_hippie.zip"

# Unzip 
unzip -o "$script_path/synthetic_joyfulyell_hippie.zip" -d $script_path

# remove zip file
rm -rf "$script_path/synthetic_joyfulyell_hippie.zip"

# Down real-world data set backpack
wget -P $script_path "https://vision.in.tum.de/webshare/u/haefner/GeneralUPS/xtion_backpack_sf4_ups.zip"

# Unzip 
unzip -o "$script_path/xtion_backpack_sf4_ups.zip" -d $script_path

# remove zip file
rm -rf "$script_path/xtion_backpack_sf4_ups.zip"

