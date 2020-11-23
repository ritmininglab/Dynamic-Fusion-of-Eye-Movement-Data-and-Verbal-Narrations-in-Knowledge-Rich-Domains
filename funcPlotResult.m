
function [] = funcPlotResult(response, num_pattern, match1, match2, mix_meta)



plotrow = 10;
plotcol = 10;

figure;
set(gcf, 'Position',  [100, 100, 700, 700]);
set(gca, 'Position',  [.0 .0 .95 .95]);
for p=1:num_pattern
    for mix=1:mix_meta(p)
        subplot(plotrow,plotcol,(p-1)*plotcol+mix);
        temp = (match1==p).*(match2==mix);
        response_p = response(temp==1,:);
        plotdata = response_p';
        markersize = 5;
        scatter(plotdata(1,:), plotdata(2,:), markersize);
        xlim([-8 8]);
        ylim([-8 8]);
        xticks([]);
        yticks([]);
        text(-5, -10, ['M',num2str(p),' S',num2str(mix)],'FontSize',15);
    end
end
a = axes;
a.Visible = 'off'; 


end
