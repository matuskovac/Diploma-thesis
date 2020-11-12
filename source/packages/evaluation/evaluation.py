from behalearn.metrics import fmr_score
from behalearn.metrics import fnmr_score
from behalearn.metrics import eer_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from shapely.geometry import LineString, Point
import  matplotlib.pyplot as plt
from packages.processing import postprocess

def show_results(test_y, predicted_y):
    print('Accuracy:', accuracy_score(test_y, predicted_y))
    print('F1 score:', f1_score(test_y, predicted_y, pos_label=1))
    print('Recall:', recall_score(test_y, predicted_y, pos_label=1))
    print('Precision:', precision_score(test_y, predicted_y, pos_label=1))
    print('\n confussion matrix:\n', confusion_matrix(test_y, predicted_y))


def plot_far_eer(test_y_raw, tresholds, selected_owners):
    tresholds, test_y_raw = zip(*sorted(zip(tresholds, test_y_raw)))
    fmr_array = []
    fnmr_array = []
    for treshold in tresholds:
        test_y, predicted_y = postprocess.unify_y_column_format(
            test_y_raw, tresholds, selected_owners, treshold)

        fmr_array.append(fmr_score(test_y, predicted_y))
        fnmr_array.append(fnmr_score(test_y, predicted_y))

#     point = eer_score(list(tresholds), fmr_array, fnmr_array)

    line1 = LineString(list(zip(tresholds, fmr_array)))
    line2 = LineString(list(zip(tresholds, fnmr_array)))

    int_pt = line1.intersection(line2)

    plt.plot(tresholds, fmr_array, 'r')  # plotting t, a separately
    plt.plot(tresholds, fnmr_array, 'b')  # plotting t, b separately
    plt.plot(int_pt.x, int_pt.y, marker='o', markersize=5, color="green")
    plt.show()

    print("EER: "+str(int_pt.y))


def get_eer(test_y_raw, tresholds, selected_owners):

    tresholds, test_y_raw = zip(*sorted(zip(tresholds, test_y_raw)))

    fmr_array = []
    fnmr_array = []
    for treshold in tresholds:
        test_y, predicted_y = postprocess.unify_y_column_format(
            test_y_raw, tresholds, selected_owners, treshold)

        fmr_array.append(fmr_score(test_y, predicted_y))
        fnmr_array.append(fnmr_score(test_y, predicted_y))

    if(all(x == 0.0 for x in tresholds)):
        return 0

    line1 = LineString(list(zip(tresholds, fmr_array)))
    line2 = LineString(list(zip(tresholds, fnmr_array)))

    int_pt = line1.intersection(line2)

    return int_pt.y
