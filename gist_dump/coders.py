# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Vlad Niculae <vlad@vene.ro>
#          Kyle Kastner <kastnerkyle@gmail.com>
#          Samantha Massengill <sgmassengill@gmail.com>
# 
# License: BSD 3 clause

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import array2d, as_float_array
from sklearn.decomposition.dict_learning import SparseCoder, sparse_encode
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from scipy.linalg import norm
import copy

#preproc.py will be a separate gist eventually
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
from sklearn.utils import array2d, as_float_array
class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples + \
                         self.bias * np.identity(n_features))
        components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        self.components_ = components
        #Order the explained variance from greatest to least
        self.explained_variance_ = eigs[::-1]
        return self

    def transform(self, X):
        X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.components_)
        return X_transformed

def local_contrast_normalization(X):
    """Normalize the patch-wise variance of the signal.

    Parameters
    ----------
    X: array-like, shape n_samples, n_features
        Data to be normalized

    Returns
    -------
    X:
        Data after individual normalization of the samples
    """
    # XXX: this should probably be extracted somewhere more general
    # center all colour channels together
    X = X.reshape((X.shape[0], -1))
    X -= X.mean(axis=1)[:, None]

    X_std = X.std(axis=1)
    # Cap the divisor to avoid amplifying samples that are essentially
    # a flat surface into full-contrast salt-and-pepper garbage.
    # the actual value is a wild guess
    # This trick is credited to N. Pinto
    min_divisor = (2 * X_std.min() + X_std.mean()) / 3
    X /= np.maximum(min_divisor, X_std).reshape(
        (X.shape[0], 1))
    return X


class RandomDataCoder(SparseCoder):
    def __init__(self, n_atoms,
                 verbose=False,
                 random_seed=None):
        self.n_atoms=n_atoms
        self.random_seed=random_seed
        self.verbose=verbose
        self.random_state = np.random.RandomState(random_seed)


    def fit(self, X, y=None, **kwargs):
        """Fit the encoder on a collection of data, e.g. image patches.

        Parameters
        ----------
        X: array-like, shape: n_samples, n_features
            the patch data to be fitted

        Returns
        -------
        self: object
            Returns the object itself
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape

        if self.verbose:
            print "About to extract atoms from %d samples" % n_samples

        #Get a random subset of length n_atoms from the input data X.
        #Each row selected via random_state.choice becomes an atom.
        indices = self.random_state.choice(np.arange(n_samples),
                                           size=self.n_atoms,
                                           replace=False)
        self.components_ = copy.copy(X[indices, :])
        return self


class KMeansCoder(SparseCoder):
    """K-means based dictionary learning

    The fit method receives an array of signals, whitens them using
    a PCA transform and run a KMeans algorithm to extract the patch
    cluster centers.

    The input is then correlated with each individual "patch-center"
    treated as a convolution kernel. The transformation can be done
    using various sparse coding methods, tresholding or the triangle
    k-means non-linearity.

    This estimator only implements the unsupervised feature extraction
    part of the referenced paper. Image classification can then be
    performed by training a linear SVM model on the output of this
    estimator.

    Parameters
    ----------
    n_atoms: int,
        number of centers extracted by the kmeans algorithm

    n_components: int, optional: default None
        number of components to keep after whitening individual samples

    max_iter: int, default 100
        maximum number of iterations to run the k-means algorithm

    n_init, int, default 1
        number of times to initialize the k-means algorithm in order to
        avoid convergence to local optima

    n_prefit: int, default 5
        dimension of reduced curriculum space in which to prefit the k-means
        algorithm for increased performance.
        This is used only when `whiten=True`.

    tol: float, default 1e-4
        tolerance for numerical errors

    verbose: bool, default False
        whether to display verbose output

    transform_algorithm: {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        Algorithm used to transform the data.
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection X.T * Y

    transform_n_nonzero_coefs: int, 0.1 * n_features by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case.

    transform_alpha: float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.

    split_sign: bool, default False
        whether to split the transformed feature vectors into positive and
        negative components, such that the downstream classification algorithms
        can assign different weights depending on the sign

    n_jobs: int,
        number of parallel jobs to run

    Attributes
    ----------
    components_: array of shape n_atoms, n_features
        centers extracted by k-means from the patch space

    Reference
    ---------
    An Analysis of Single-Layer Networks in Unsupervised Feature Learning
    Adam Coates, Honglak Lee and Andrew Ng. In NIPS*2010 Workshop on
    Deep Learning and Unsupervised Feature Learning.
    http://robotics.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf

    """
    def __init__(self, n_atoms, n_components=None,
                 max_iter=100, tol=1e-4,
                 verbose=False,
                 n_init=1,
                 transform_algorithm='omp',
                 transform_n_nonzero_coefs=None,
                 n_jobs=1):
        self.n_atoms = n_atoms
        self.max_iter = max_iter
        self.n_components = n_components
        self.n_init = n_init
        self.verbose = verbose
        self.tol = tol
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.n_jobs = n_jobs


    def fit(self, X, y=None, **kwargs):
        """Fit the encoder on a collection of data, e.g. image patches.

        Parameters
        ----------
        X: array-like, shape: n_samples, n_features
            the patch data to be fitted

        Returns
        -------
        self: object
            Returns the object itself
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        # kmeans model to find the filters
        if self.verbose:
            print "About to extract atoms from %d samples" % n_samples
        kmeans = KMeans(n_clusters=self.n_atoms, init='k-means++',
                        max_iter=self.max_iter, n_init=self.n_init,
                        tol=self.tol, verbose=self.verbose)

        kmeans.fit(X)
        self.components_ = kmeans.cluster_centers_

        self.kmeans = kmeans
        self.inertia_ = kmeans.inertia_
        return self


class KSVDCoder(SparseCoder):
    def __init__(self, n_atoms,
                 verbose=False,
                 approximate=True,
                 transform_algorithm='threshold',
                 transform_alpha=0,
                 n_iter=1,
                 random_seed=None):
        self.n_atoms = n_atoms
        self.random_seed = random_seed
        self.approximate = approximate
        self.verbose = verbose
        self.n_iter = n_iter
        self.transform_algortihm = transform_algorithm
        self.transform_alpha = transform_alpha
        self.random_state = np.random.RandomState(random_seed)


    def fit(self, X, y=None, **kwargs):
        """Fit the encoder on a collection of data, e.g. image patches.

        Parameters
        ----------
        X: array-like, shape: n_samples, n_features
            the patch data to be fitted

        Returns
        -------
        self: object
            Returns the object itself
        """
        X = np.atleast_2d(X)
        X -= np.mean(X, axis=0)
        n_samples, n_features = X.shape

        if self.verbose:
            print("About to extract atoms from %d samples" % n_samples)
            print("Input array is %d by %d" % (n_samples, n_features))

        #Initialize a dictionary in sklearn style - rows are samples
        #Make sure a data entry has some content
        #Could make this cleaner by only calculating column norm for used
        usable_entries = list(np.where(np.sum(X**2., axis=1) > 1E-6)[0])
        subset = self.random_state.choice(usable_entries, size=self.n_atoms,
                                          replace=False)
        #Normalize atoms in dictionary
        #Doing it without norm and axis argument for backwards compatibility...
        #See http://stackoverflow.com/questions/7741878/ \
        #        how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        D = X[subset, :] / (np.sum(np.abs(X[subset, :])**2,
                                                      axis=-1)**(1./2))[None].T
        for i in range(self.n_iter):
            if self.verbose:
                print("Performing iter %d" % i)
            W = sparse_encode(X, D,
                              algorithm='omp',
                              n_nonzero_coefs=None)

            residual = R = X - np.dot(W, D)
            for k in range(self.n_atoms):
                if self.approximate:
                    R, W, D = self._approximateSVD(k, R, W, D)
                else:
                    R, W, D = self._exactSVD(k, R, W, D)
        self.components_ = D
        return self


    def _exactSVD(self, k, R, W, D):
        #For more information, visit
        #http://www.ux.uis.no/~karlsk/dle/#ssec32
        I = np.nonzero(W[:, k])[0]
        #Need these slices to stay as matrices
        project = P = np.dot(W[I, k][:, None], D[k, :][:, None].T)
        Ri = R[I, :] + P
        #Transpose to make the svds work correctly
        U, S, V = svds(Ri.T, 1)
        D[k, :] = U.ravel()
        W[I, k] = np.dot(V.T, S)
        R[I, :] = Ri - P
        return R, W, D


    def _approximateSVD(self, k, R, W, D):
        #For more information, visit
        #http://www.ux.uis.no/~karlsk/dle/#ssec32
        I = np.nonzero(W[:, k])[0]
        #Need these slices to stay as matrices
        project = P = np.dot(W[I, k][:, None], D[k, :][:, None].T)
        Ri = R[I, :] + P
        dk = np.dot(Ri.T, W[I, k][:, None])
        dk /= norm(dk)
        D[k, :] = dk.ravel()
        W[I, k] = np.dot(Ri, dk).ravel()
        R[I, :] = Ri - P
        return R, W, D


if __name__ == '__main__':
    """
    Dictionary learning with K-Means on faces image data
    ====================================================

    This shows dictionary atoms learned from image patches extracted from
    the face recognition dataset. The dictionary atoms are learned using
    (:ref:`KMeansCoder`), with and respectively without a whitening PCA transform.

    The dataset used in this example is a preprocessed excerpt of the
    "Labeled Faces in the Wild", aka LFW_:

    http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

    .. _LFW: http://vis-www.cs.umass.edu/lfw/

    .. |kc_no_w| image:: /images/plot_kmeans_coder_1.png
    :scale: 50%

    .. |kc_w| image:: /images/plot_kmeans_coder_2.png
    :scale: 50%

    .. centered:: |kc_no_w| |kc_w|

    """
    print __doc__

    from time import time
    import logging
    import pylab as pl

    import numpy as np

    from sklearn.cross_validation import StratifiedKFold
    from sklearn.datasets import fetch_lfw_people
    from sklearn.feature_extraction.image import PatchExtractor

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    ###########################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # reshape the data using the traditional (n_samples, n_features) shape
    faces = lfw_people.data
    n_samples, h, w = lfw_people.images.shape

    X = faces
    X -= X.mean(axis=1)[:, np.newaxis]
    n_features = X.shape[1]
    X = X.reshape((n_samples, h, w))

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print "Total dataset size:"
    print "n_samples: %d" % n_samples
    print "n_features: %d" % n_features
    print "n_classes: %d" % n_classes


    ###########################################################################
    # Split into a training set and a test set using a stratified k fold

    train, test = iter(StratifiedKFold(y, n_folds=4)).next()
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]


    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction

    #TODO: Support for "non-square" numbers of atoms to display
    #This number should be square i.e. 16, 25, 36, 49, 64 etc.
    n_atoms = 144
    patch_size = (8,8)

    print "Extracting image patches from %d faces" % len(X_train)
    t0 = time()
    extr = PatchExtractor(patch_size=patch_size,
                          max_patches=100, random_state=0)
    patches = extr.transform(X_train)
    print "done in %0.3fs" % (time() - t0)

    print "Extracting %d atoms from %d patches" % (
        n_atoms, len(patches))
    t0 = time()
    patches = patches.reshape((patches.shape[0],
                               patches.shape[1] * patches.shape[2]))

    sc1 = KSVDCoder(n_atoms, verbose=True, n_iter=5)
    rc1 = RandomDataCoder(n_atoms)
    kc1 = KMeansCoder(n_atoms, verbose=True)
    steps = [('pre', ZCA()),
             ('dict', kc1)]
    p_kmeans = Pipeline(steps)
    p_kmeans.fit(patches)
    print "done in %0.3fs" % (time() - t0)

    t0 = time()
    steps = [('pre', ZCA()),
             ('dict', sc1)]
    p_ksvd = Pipeline(steps)
    p_ksvd.fit(patches)
    print "done in %0.3fs" % (time() - t0)

    #print "Extracting %d whitened atoms from %d patches" % (
    #    n_atoms, len(patches))
    #t0 = time()
    #print "done in %0.3fs" % (time() - t0)

    ###############################################################################
    # Qualitative evaluation of the extracted filters

    n_row = int(np.sqrt(n_atoms))
    n_col = int(np.sqrt(n_atoms))
    pipelines = [p_kmeans,
                 p_ksvd]

    for pipe in pipelines:
        pl.figure(figsize=(5, 6))
        title = ""
        for alg_tup in pipe.steps:
            title += '%s,'%(alg_tup[-1].__class__.__name__)
        pl.suptitle("Dictionary learned on the \n LFW dataset with " + title)
        for i, atom in enumerate(pipe.named_steps['dict'].components_):
            pl.subplot(n_row, n_col, i + 1)
            pl.imshow(atom.reshape(patch_size), cmap=pl.cm.gray,
                                            interpolation="nearest")
            pl.xticks(())
            pl.yticks(())
        pl.subplots_adjust(0.02, 0.03, 0.98, 0.90, 0.14, 0.01)
    pl.show()
