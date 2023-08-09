function prior = computeprior(pars,parnames)

    figure
for i=1:size(pars,2)
    subplot(1,size(pars,2),i)

    h= histfit(pars(:,i),20,'kernel'); % 20 = num bins
    title(parnames(i))
    prior{i} = [h(2).XData;h(2).YData];
end
end