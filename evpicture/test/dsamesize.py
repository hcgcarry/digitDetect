 ls -l|grep '^-'|awk '{if(a[$5]){ a[$5]=a[$5]"\n"$NF; b[$5]++;} else a[$5]=$NF} END{for(x in b)print a[x];}'

