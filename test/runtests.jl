using Test, ForecastVerification


@testset "Brier score" begin
y_true = [  1,  0,  1,  1,  1,  0]
y_prob = [0.9,0.1,0.9,0.9,0.1,0.1]


#import sklearn.metrics
#sklearn.metrics.brier_score_loss(y_true,y_prob)

@test ForecastVerification.brierscore(y_true,y_prob) ≈ 0.14333333333333334

BS2,REL,RES,UNC = ForecastVerification.brierscoredecom(y_true,y_prob)
@test BS2 ≈ ForecastVerification.brierscore(y_true,y_prob)


end

@testset "CRPS" begin
    # test data and reference values are from sangoma
    # https://sourceforge.net/projects/sangoma/files/

    X = [1.1   0.8   2.2   0.5   2.4;
         2.3   1.1   3.0   0.7   0.7;
         1.3   3.2   0.7   2.2   3.1;
         0.8   0.5   1.5   2.4   0.4]
    xobs = [ 0.78, 2.47, 3.21, 1.13 ];
    @test ForecastVerification.CRPS(X,xobs) ≈ 0.44350
end
