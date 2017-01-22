//
//  LandingViewController.swift
//  FaceCreeper
//
//  Created by Austin Han on 2017-01-21.
//  Copyright © 2017 Hackathon. All rights reserved.
//

import UIKit

class LandingViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {

    @IBOutlet weak var labelView: UILabel!
    @IBOutlet weak var tableView: UITableView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var descrView: UILabel!
    var id = -1
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.clipsToBounds = true;
        imageView.layer.cornerRadius = 20
        imageView.layer.borderColor = UIColor.white.cgColor
        imageView.layer.borderWidth = 5
        
        tableView.delegate = self
        tableView.dataSource = self
        switch (id){
        case -1:
            self.imageView.image = UIImage(named: "melissa")
            labelView.text = "Melissa Ng"
            descrView.text = "21, Female, Iron(FE) Designer, UX Research Assistant"
        case 0:
            self.imageView.image = UIImage(named: "angie")
            labelView.text = "Angie Harmon"
            descrView.text = "44, Female, Actress"
        case 1:
            self.imageView.image = UIImage(named: "daniel")
            labelView.text = "Daniel Radcliffe"
            descrView.text = "27, Male, Actor"
        case 2:
            self.imageView.image = UIImage(named: "samie")
            labelView.text = "Farhan Samir"
            descrView.text = "21, Male, Software Developer"
        case 3:
            self.imageView.image = UIImage(named: "lorainne")
            labelView.text = "Lorraine Bracco"
            descrView.text = "62, Female, Actress"
        case 4:
            self.imageView.image = UIImage(named: "michael")
            labelView.text = "Michael Varton"
            descrView.text = "48, Male, Actor"
        case 5:
            self.imageView.image = UIImage(named: "peri")
            labelView.text = "Peri Gilpin"
            descrView.text = "55, Male, Actor"
        default:
            self.imageView.image = UIImage(named: "melissa")
            labelView.text = "Melissa Ng"
            descrView.text = "21, Female, Iron(FE) Designer, UX Research Assistant"
        }
        // Do any additional setup after loading the view.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    

    
     func numberOfSections(in tableView: UITableView) -> Int {
        // #warning Incomplete implementation, return the number of sections
        return 1
    }
    
     func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        // #warning Incomplete implementation, return the number of rows
        return 3
    }
    
    
     func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "NetworkTableViewCell", for: indexPath) as! NetworkTableViewCell
        
        switch (indexPath.row){
        case 0:
            cell.imageView?.image = UIImage(named: "git")
            if (id == 2){ // Farhan Samir
                cell.NetworkLabel.text = "https://www.github.com/fsam95"
            } else {
                cell.NetworkLabel.text = "https://www.github.com"
            }
            break
        case 1:
            cell.imageView?.image = UIImage(named: "facebook")
            if (id == 2){ // Farhan Samir
                cell.NetworkLabel.text = "https://www.facebook.com/farhan.samir.1"
            } else {
                cell.NetworkLabel.text = "https://www.facebook.com"
            }
            break
        case 2:
            cell.imageView?.image = UIImage(named: "twitter")
            if (id == 2){ // Farhan Samir
                cell.NetworkLabel.text = "https://twitter.com/farhan_samir"
            } else {
                cell.NetworkLabel.text = "https://www.twitter.com"
            }
            break
        default:
            return cell;
        }
        return cell
        
    }
    
     func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 60
        
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
