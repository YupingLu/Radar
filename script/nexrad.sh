#!/bin/bash

# Shell script to extract specific files from nexrad level III data
# Author: Yuping Lu
# Date: 01/14/2019

search_dir='/Users/ylk/Downloads/nexrad/HAS011208157'

N0R_dir='/Users/ylk/Downloads/nexrad/processed/N0R'
N0X_dir='/Users/ylk/Downloads/nexrad/processed/N0X'
N0C_dir='/Users/ylk/Downloads/nexrad/processed/N0C'
N0K_dir='/Users/ylk/Downloads/nexrad/processed/N0K'
N0H_dir='/Users/ylk/Downloads/nexrad/processed/N0H'

target_dir='/Users/ylk/Downloads/nexrad/processed/final'

# remove files in target dir and clear n0r.txt n0x.txt n0c.txt n0k.txt n0h.txt
`rm "${target_dir}"/*`
> n0r.txt
> n0x.txt
> n0c.txt
> n0k.txt
> n0h.txt

for entry in "$search_dir"/*
do
    # extract files from tar.gz files
    tar -xzf "$entry" -C "$N0R_dir" --wildcards '*N0R*'
    tar -xzf "$entry" -C "$N0X_dir" --wildcards '*N0X*'
    tar -xzf "$entry" -C "$N0C_dir" --wildcards '*N0C*'
    tar -xzf "$entry" -C "$N0K_dir" --wildcards '*N0K*'
    tar -xzf "$entry" -C "$N0H_dir" --wildcards '*N0H*'
    
    # match N0H with other files and move them to final
    entries=`ls $N0H_dir`
    counter=0
    for entry in $entries
    do
        if [ $(($counter%1)) -eq 0 ]; then
            pattern="*${entry: -12}"
            if [[ `find $N0R_dir -name $pattern` && `find $N0X_dir -name $pattern` && `find $N0C_dir -name $pattern` && `find $N0K_dir -name $pattern` ]]; then
                echo "$entry" >> n0h.txt
                mv "${N0H_dir}/${entry}" "$target_dir"
                
                f1=`find $N0R_dir -name $pattern`
                n0r=$(basename ${f1[0]})
                echo "$n0r" >> n0r.txt
                mv "${N0R_dir}/${n0r}" "$target_dir"
    
                f2=`find $N0X_dir -name $pattern`
                n0x=$(basename ${f2[0]})
                echo "$n0x" >> n0x.txt
                mv "${N0X_dir}/${n0x}" "$target_dir"
    
                f3=`find $N0C_dir -name $pattern`
                n0c=$(basename ${f3[0]})
                echo "$n0c" >> n0c.txt
                mv "${N0C_dir}/${n0c}" "$target_dir"
                
                f4=`find $N0K_dir -name $pattern`
                n0k=$(basename ${f4[0]})
                echo "$n0k" >> n0k.txt
                mv "${N0K_dir}/${n0k}" "$target_dir"
            fi
        fi
        counter=$((counter+1))
    done
    
    # remove files in N0R etc.
    `rm "${N0R_dir}"/*`
    `rm "${N0X_dir}"/*`
    `rm "${N0C_dir}"/*`
    `rm "${N0K_dir}"/*`
    `rm "${N0H_dir}"/*`
done
