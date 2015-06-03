
% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'YGrid','on');
%% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[0 17000]);
box(axes1,'on');
hold(axes1,'on');

% Create errorbar

errorbar(wikimean(:,1), wikimean(:,2), wikimean(:,3),'DisplayName','wiki\_mean');

errorbar(wikihidsemi(:,1), wikihidsemi(:,2), wikihidsemi(:,3),'DisplayName','wiki\_semi');

errorbar(mean(:,1), mean(:,2), mean(:,3),'DisplayName','mean');

errorbar(nmf(:,1), nmf(:,2), nmf(:,3),'DisplayName','nmf');

%errorbar(wikihid(:,1), wikihid(:,2), wikihid(:,3),'DisplayName','wiki\_hnorm');

%errorbar(wikiall(:,1), wikiall(:,2), wikiall(:,3),'DisplayName','wiki\_allnorm');

%errorbar(norm(:,1), norm(:,2), norm(:,3),'DisplayName','allnorm');

%errorbar(nonorm(:,1), nonorm(:,2), nonorm(:,3),'DisplayName','nonorm');

%errorbar(nosemi(:,1), nosemi(:,2), nosemi(:,3),'DisplayName','nosemi');



% Create xlabel
xlabel({'Training sentences'});

% Create ylabel
ylabel({'Classification accuracy'});

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.5631952599538 0.555699864499463 0.217758982239515 0.21892654798967]);

