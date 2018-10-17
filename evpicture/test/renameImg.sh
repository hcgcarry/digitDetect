#! /bin/bash
i=0
for img in $(ls|grep jpg);do
	let i++
	mv $img "jpg${i}.jpg" 
done
let i=0
echo "$i"
find -name "* *" -type f | rename 's/ /_/g'
for img in $(find -name "*.jpeg" -type f);do
	let i++   
	mv $img "ujpeg${i}.jpeg"
done
