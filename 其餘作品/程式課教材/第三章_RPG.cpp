#include<bits/stdc++.h>
using namespace std;
int main(){
	srand(time(0));
	int BHP=10000;
	int BATK=200;
	int PHP=1000;
	int PATK=100;
	string Bskill[3]={"�ޤM��","��q����","�J�@"};
	string Pskill[3]={"��ŧ","�C��","�~�X��"};
	for(;;){
		
		cout<<"Boss HP:"<<BHP<<endl;
		cout<<"Player HP:"<<PHP<<endl;
		cout<<endl<<"���q����[A] / �ޯ�[S] / �^�_[D] / �k�][F] :";
		char choise;cin>>choise; 
		cout<<endl<<"---------------------------------"<<endl;
		if(choise=='A'){
			cout<<"�A��Boss�ϥδ��q�����A�y��"<<PATK<<"���ˮ`"<<endl; 
			BHP=BHP-PATK; 
		}
		if(choise=='S'){
			cout<<"��ŧ[A] / �C��[S] / �~�X��[D] :";
			char skill;cin>>skill;
			if(skill=='A'){
				cout<<"��ܰ�ŧ!"<<endl; 
				if(rand()%100<50){
					cout<<"��ŧ���\�A�R���z�I�y��"<<PATK*2<<"���ˮ`"<<endl; 
					BHP=BHP-PATK*2;
				}
				else cout<<"��ŧ���ѤF..."<<endl; 
			}
			if(skill=='S'){
				cout<<"��Boss�٥X�C��"<<endl; 
				BHP=BHP-PATK*0.8;
				if(rand()%100<30){
					cout<<"�]���j�N�F�S���{�A�y��"<<PATK*2.2<<"�I�ˮ`"<<endl; 
					BHP=BHP-PATK*1.4;
				}
				else cout<<"�]���ɤU�C����٬O����"<<PATK*0.8<<"�I�ˮ`"<<endl; 
			}
			if(skill=='D'){
				cout<<"�@�ߤ@���A�~�X��"<<endl; 
				if(rand()%100<10){
					cout<<"�U�Q���@���A�y��"<<PATK*10<<"�I�ˮ`"<<endl; 
					BHP=BHP-PATK*10;
				}
				else cout<<"�ܦn�A�S������"<<endl; 
			}	
		}
		if(choise=='D'){
			cout<<"�v���ۤv"<<endl; 
			PHP=PHP+rand()%300+200;	
		} 
		if(choise=='F'){
			cout<<"�A��ܰk�]...";
			return 0;
		}  
		if(rand()%100<65){
			cout<<"�]�����M�A����"<<BATK<<"�I�ˮ`"<<endl;
			PHP=PHP-BATK; 
		}
		else {
			int skill=rand()%3;
			if(skill==1){
				cout<<"�]���ϥX�ޤM�١A�y��"<<BATK+PHP*0.05<<"�I�ˮ`"<<endl;
				PHP=PHP-BATK-PHP*0.05;
			}	
			if(skill==2){
				cout<<"�]�������A����G�A�l��"<<BATK+BHP*0.01<<"�I"<<endl;
				BHP=BHP+BATK+BHP*0.01;
				PHP=PHP-BATK-BHP*0.01; 
			}
			if(skill==3){
				cout<<"�]���o�ʡi���@�j�A�H�崫��";
				BHP=BHP*0.5;
				PHP=PHP-BATK*BHP/1000;
				cout<<"....����"<<BATK*BHP/1000<<"�I�ˮ`"<<endl; 
			}
		}
		if(BHP<=0){
			cout<<"�]�����ѡA�AĹ�F"<<endl;
			break; 
		}
		if(PHP<=0){
			cout<<"�̫��٬O�L�k�s��..."<<endl;
			break; 
		}
	} 
} 
