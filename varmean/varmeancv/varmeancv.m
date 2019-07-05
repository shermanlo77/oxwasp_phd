experiments = {...
  VarMeanCvAbsNoFilterNull();
  VarMeanCvAbsNoFilterBw();
  VarMeanCvAbsNoFilterLinear();
  VarMeanCvAbsFilterNull();
  VarMeanCvAbsFilterBw();
  VarMeanCvAbsFilterLinear();
};

for i = 1:numel(experiments)
  experiments{i}.run();
end