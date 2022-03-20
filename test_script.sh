for n in */*.ipynb
do
    if [[ "$n" != "submission-example-1/clustering_using_riemannian_mean_shift_algorithm.ipynb" ]]
    then
            poetry run jupyter nbconvert --to notebook --execute $n
    else
            echo "No notebooks found!"
    fi
done