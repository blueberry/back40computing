#!/bin/sh

OPTIONS="--i=100 --src=randomize --device=1 --instrumented"
SUFFIX="instrumented"

for i in audikw1.graph cage15.graph coPapersCiteseer.graph europe.osm.graph hugebubbles-00020.graph kkt_power.graph kron_g500-logn20.graph 
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../../graphs/$i $OPTIONS  
	./bin/test_bfs_4.0_i386 metis ../../../../graphs/$i $OPTIONS > eval/$i.$SUFFIX.txt
	sleep 5 
done

for i in nlpkkt160.graph 
do
	echo ./bin/test_bfs_4.0_i386 metis ../../../../graphs/$i $OPTIONS --queue-sizing=0.5 
	./bin/test_bfs_4.0_i386 metis ../../../../graphs/$i $OPTIONS --queue-sizing=0.5 > eval/$i.$SUFFIX.txt 
	sleep 5 
done

for i in wikipedia-20070206.mtx
do
	echo ./bin/test_bfs_4.0_i386 market ../../../../graphs/$i $OPTIONS 
	./bin/test_bfs_4.0_i386 market ../../../../graphs/$i $OPTIONS > eval/$i.$SUFFIX.txt 
	sleep 5 
done

echo /bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.7 $OPTIONS 
./bin/test_bfs_4.0_i386 grid2d 5000 --queue-sizing=0.7 $OPTIONS > eval/grid2d.5000.$SUFFIX.txt	
	sleep 5 

echo /bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.6 $OPTIONS 
./bin/test_bfs_4.0_i386 grid3d 300 --queue-sizing=0.6 $OPTIONS > eval/grid3d.300.$SUFFIX.txt	
	sleep 5 

i=random.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../../graphs/$i $OPTIONS 
./bin/test_bfs_4.0_x86_64 dimacs ../../../../graphs/$i $OPTIONS > eval/$i.$SUFFIX.txt 
	sleep 5 
 
i=rmat.2Mv.128Me.gr
echo ./bin/test_bfs_4.0_x86_64 dimacs ../../../../graphs/$i $OPTIONS 
./bin/test_bfs_4.0_x86_64 dimacs ../../../../graphs/$i $OPTIONS > eval/$i.$SUFFIX.txt 
	sleep 5 
