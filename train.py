from DatasetManager.Participant import Participant

from DatasetManager.Enums.Day import Day
from DatasetManager.Enums.MileStone import MileStone


def main():
    part = Participant(10376)


    #filtered_results = pl.ml.filter_csv_data(pl.ml.metadata, "MFISTO1N", ">", 0)
    #questionaire_results = pl.ml.get_csv_column(filtered_results, "MFISTO1N")
    #print(questionaire_results)

    json_data = part.sl.get_walking_bout_analysis_dmo(Day.DAY1, MileStone.T3)

    json_base = json_data["WBASO_Output"]["TimeMeasure1"]["Recording1"]["SU"]["LowerBack"]

    print(json_base["MetaData"])

    print(len(part.filter_json_arr(json_base["LevelWB"], "NumberStrides", "==", 5)))
    

if __name__ == "__main__":
    main()