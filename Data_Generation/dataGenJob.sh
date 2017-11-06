for i in {0..365}
do
	echo $i
	dayNum=$i
	export dayNum
	qsub run_PyDataGen.pbs
done
